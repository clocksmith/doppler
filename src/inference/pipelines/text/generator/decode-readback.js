import { FINITENESS_RESET_WORDS, parseFinitenessStatusWords } from '../finiteness-guard-status.js';

// Readback of a recorded decode step's staging buffer: the sampled token and finiteness
// words, or a plain mapped copy.

/** @type {typeof import('./decode-readback.js').readSampledTokenFromStagingBuffer} */
export async function readSampledTokenFromStagingBuffer(stagingBuffer, options = {}) {
  const ownsStagingBuffer = options.ownsStagingBuffer === true;
  const hasFinitenessBuffer = options.hasFinitenessBuffer === true;
  const ring = options.ring ?? null;
  const cleanupRecorder = options.cleanupRecorder ?? null;
  const timing = {
    mapWaitMs: 0,
    cleanupMs: 0,
    copyMs: 0,
  };
  let mapped = false;
  let cleanupCompleted = false;

  try {
    const mapStart = performance.now();
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    timing.mapWaitMs = performance.now() - mapStart;
    mapped = true;
    const cleanupStart = performance.now();
    await cleanupRecorder?.completeDeferredCleanup();
    timing.cleanupMs = performance.now() - cleanupStart;
    cleanupCompleted = true;
    const copyStart = performance.now();
    const mappedWords = new Uint32Array(stagingBuffer.getMappedRange());
    if (mappedWords[0] === 0xFFFFFFFF) throw new Error('[Sampling] No finite candidate logits.');
    const result = {
      nextToken: mappedWords[0],
      finitenessStatus: hasFinitenessBuffer
        ? parseFinitenessStatusWords(mappedWords, 1)
        : parseFinitenessStatusWords(FINITENESS_RESET_WORDS, 0),
      timing,
    };
    timing.copyMs = performance.now() - copyStart;
    return result;
  } finally {
    if (mapped) stagingBuffer.unmap();
    try {
      if (!cleanupCompleted) {
        await cleanupRecorder?.completeDeferredCleanup({ discardPooled: true });
      }
    } finally {
      if (ownsStagingBuffer) {
        stagingBuffer.destroy();
      }
      ring?.advance();
    }
  }
}

/** @type {typeof import('./decode-readback.js').readMappedBufferCopy} */
export async function readMappedBufferCopy(stagingBuffer, options = {}) {
  const ownsStagingBuffer = options.ownsStagingBuffer !== false;
  let mapped = false;

  try {
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    mapped = true;
    return stagingBuffer.getMappedRange().slice(0);
  } finally {
    if (mapped) {
      stagingBuffer.unmap();
    }
    if (ownsStagingBuffer) {
      stagingBuffer.destroy();
    }
  }
}
