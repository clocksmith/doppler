// Terminating the worker also cancels decompression, text decoding and a running JSON.parse.
import config from './activation-grid/config.json' with { type: 'json' };

/** @type {typeof import('./recording-import.js').readRecordingJson} */
export function readRecordingJson(file, { signal, maxBytes = config.maxImportBytes } = {}) {
  signal?.throwIfAborted();
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1 || maxBytes > config.maxImportBytes) throw new Error('Invalid recording import limit.');
  if (file.size > maxBytes) throw new Error('Recording file exceeds the import limit.');
  const worker = new Worker(new URL('./recording-import-worker.js', import.meta.url), { type: 'module' });
  return new Promise((resolve, reject) => {
    function cleanup() {
      signal?.removeEventListener('abort', abort);
      worker.terminate();
    }
    const fail = (/** @type {unknown} */ error) => {
      cleanup();
      reject(error);
    };
    const abort = () => fail(signal?.reason);
    signal?.addEventListener('abort', abort, { once: true });
    worker.onmessage = ({ data }) => {
      if (data.error) return fail(new Error(data.error));
      cleanup();
      resolve(data);
    };
    worker.onerror = event => fail(new Error(event.message || 'Could not read the recording.'));
    worker.onmessageerror = () => fail(new Error('Could not transfer the recording.'));
    try {
      worker.postMessage({ file, limit: maxBytes });
    } catch (error) {
      fail(error);
    }
  });
}

// Verified built-in recordings are also capped while receiving the response body.
/** @type {typeof import('./recording-import.js').readRecordingBlob} */
export async function readRecordingBlob(response, expectedBytes) {
  if (!Number.isSafeInteger(expectedBytes) || expectedBytes < 1 || expectedBytes > config.maxImportBytes) {
    throw new Error('Invalid recording download size.');
  }
  if (!response.body) throw new Error('Incomplete recording download.');
  const reader = response.body.getReader();
  const pieces = [];
  let size = 0;
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > expectedBytes) {
        await reader.cancel();
        throw new Error('Recording exceeds the declared download size.');
      }
      pieces.push(value);
    }
    if (size !== expectedBytes) throw new Error('Incomplete recording download.');
    return new Blob(/** @type {BlobPart[]} */ (pieces));
  } finally { reader.releaseLock(); }
}
