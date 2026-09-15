import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { getKernelConfig } from './kernel-configs.js';
import { unifiedKernelWrapper } from './kernel-execution.js';

export async function recordSuppressLogits(recorder, logits, vocabSize, tokenIds) {
  if (!Number.isSafeInteger(vocabSize) || vocabSize < 1 || logits.size < vocabSize * 4) {
    throw new Error('[LogitSuppress] Invalid vocabulary or logits buffer size.');
  }
  if (!Array.isArray(tokenIds) || tokenIds.some(id => !Number.isInteger(id) || id < 0 || id >= 0xFFFFFFFF)) {
    throw new Error('[LogitSuppress] Suppression requires unsigned token IDs.');
  }
  const ids = Uint32Array.from(new Set(tokenIds.filter(id => id < vocabSize)));
  if (!ids.length) return;
  const buffer = acquireBuffer(ids.byteLength, undefined, 'sampling_suppressed_ids');
  let transferred = false;
  try {
    recorder.trackTemporaryBuffer(buffer);
    transferred = true;
    recorder.device.queue.writeBuffer(buffer, 0, ids);
    const config = getKernelConfig('logit_suppress', 'f32');
    await unifiedKernelWrapper('logit_suppress', recorder, 'f32', [buffer, logits],
      { vocab_size: vocabSize, token_count: ids.length }, Math.ceil(ids.length / config.workgroupSize[0]));
  } finally { if (!transferred) releaseBuffer(buffer); }
}
