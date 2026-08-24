import { readBuffer } from '../../../../memory/buffer-pool.js';

export async function readBufferWithCleanup(buffer, byteLength, cleanup, reader = readBuffer) {
  try {
    return await reader(buffer, byteLength);
  } finally {
    cleanup?.();
  }
}
