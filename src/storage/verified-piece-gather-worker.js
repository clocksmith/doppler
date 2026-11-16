import { parentPort, workerData } from 'node:worker_threads';
import { openSync, readSync, closeSync } from 'node:fs';
import { createVerifiedPieceReader } from './verified-piece-reader.js';

// This worker owns acquisition, complete-piece authentication and range gathering.
// Only the finished range crosses the thread boundary. No caller scratch is detached.
/** @type {Map<string, number>} */
const handles = new Map();
if (!parentPort) throw new Error('Gather requires a worker port.');
const port = parentPort;
const reader = await createVerifiedPieceReader({ ...workerData.metadata,
  acquire: async piece => {
    const filename = workerData.files[piece.path];
    if (typeof filename !== 'string') throw new Error('Missing approved piece file.');
    let handle = handles.get(piece.path);
    if (handle === undefined) {
      handle = openSync(filename, 'r');
      handles.set(piece.path, handle);
    }
    const bytes = new Uint8Array(piece.size);
    let read = 0;
    while (read < bytes.length) {
      const count = readSync(handle, bytes, read, bytes.length - read, piece.offset + read);
      if (!count) throw new Error('Incomplete authenticated piece.');
      read += count;
    }
    return bytes;
  },
});
port.postMessage({ ready: true });
port.on('message', async ({ path, offset, length, buffer }) => {
  try {
    await reader.readInto(path, offset, new Uint8Array(buffer, 0, length));
    port.postMessage({ buffer, length }, [buffer]);
  } catch (error) {
    // Never publish a verified prefix after a later piece fails verification.
    new Uint8Array(buffer).fill(0);
    port.postMessage({ buffer, error: String(error.message ?? error) }, [buffer]);
  }
});
process.on('exit', () => { for (const handle of handles.values()) closeSync(handle); });
