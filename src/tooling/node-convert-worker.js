import { parentPort } from 'node:worker_threads';
import { transformTensorBytes } from '../converter/core.js';

function toOwnedUint8Array(bytes) {
  if (!(bytes instanceof Uint8Array)) {
    throw new Error('Worker transform must return Uint8Array bytes.');
  }
  if (bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength) {
    return bytes;
  }
  return bytes.slice();
}

if (!parentPort) {
  throw new Error('node-convert-worker requires parentPort.');
}

parentPort.on('message', (message) => {
  const id = message?.id;
  const job = message?.job;
  try {
    if (!Number.isInteger(id) || id < 1) {
      throw new Error('Worker message id must be a positive integer.');
    }
    if (!job || typeof job !== 'object') {
      throw new Error('Worker job payload must be an object.');
    }
    const tensor = job.tensor;
    const rawData = job.tensorData;
    if (!tensor || typeof tensor !== 'object') {
      throw new Error('Worker job tensor is required.');
    }
    if (!(rawData instanceof ArrayBuffer)) {
      throw new Error('Worker job tensorData must be an ArrayBuffer.');
    }

    const transformed = transformTensorBytes(tensor, new Uint8Array(rawData), job.transformContext ?? {});
    const outBytes = toOwnedUint8Array(transformed.tensorData);
    const companionBytes = transformed.companionData instanceof Uint8Array
      ? toOwnedUint8Array(transformed.companionData)
      : null;
    const transferList = [outBytes.buffer];
    if (companionBytes && companionBytes.buffer !== outBytes.buffer) {
      transferList.push(companionBytes.buffer);
    }
    parentPort.postMessage({
      id,
      ok: true,
      result: {
        tensorData: outBytes.buffer,
        outDtype: transformed.outDtype ?? tensor.dtype ?? null,
        outLayout: transformed.outLayout ?? null,
        ...(companionBytes ? { companionData: companionBytes.buffer } : {}),
        ...(transformed.sourceTransform ? { sourceTransform: transformed.sourceTransform } : {}),
      },
    }, transferList);
  } catch (error) {
    const messageText = error instanceof Error ? error.message : String(error);
    const stack = error instanceof Error ? error.stack : null;
    parentPort.postMessage({
      id,
      ok: false,
      error: {
        message: messageText,
        stack,
      },
    });
  }
});
