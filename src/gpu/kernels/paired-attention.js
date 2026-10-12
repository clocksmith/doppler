import { createTensor } from '../tensor.js';
import { getDevice } from '../device.js';
import { unifiedKernelWrapper, withKernelOutput } from './kernel-execution.js';
import { getKernelPathAttentionVariant } from '../../config/kernel-path-loader.js';

/** @type {Readonly<import('../tensor.js').TensorRepresentation>} */
const PAIR = Object.freeze({ precision: 'two-f32', components: 2, elementStrideBytes: 8 });
/** @type {typeof import('./paired-attention.js').executePairedAttention} */
export async function executePairedAttention(recorder, Q, K, V, mask, numHeads, headDim, options) {
  const { seqLen, kvLen, numKVHeads, scale, causal, startPos, kvStart = 0, slidingWindow: declaredSlidingWindow,
    attnSoftcap, outputBuffer = null } = options;
  if (declaredSlidingWindow === undefined || attnSoftcap === undefined) {
    throw new Error('[PairedAttention] Explicit disabled or enabled window and softcap policy required.');
  }
  // The model contract uses null to explicitly disable the sliding window.
  const slidingWindow = declaredSlidingWindow === null ? 0 : declaredSlidingWindow;
  if (seqLen === undefined || kvLen === undefined || numKVHeads === undefined || startPos === undefined || scale === undefined) {
    throw new Error('[PairedAttention] Explicit geometry, position and scale required.');
  }
  const dimensions = [seqLen, kvLen, numHeads, numKVHeads, headDim];
  if (!dimensions.every(value => Number.isSafeInteger(value) && value > 0 && value <= 0xffffff)
    || numHeads % numKVHeads !== 0 || headDim > 256) throw new Error('[PairedAttention] Explicit positive geometry with headDim <= 256 and integral GQA required.');
  if (![startPos, kvStart, slidingWindow].every(value => Number.isSafeInteger(value) && value >= 0 && value <= 0xffffff)
    || typeof causal !== 'boolean' || !Number.isFinite(scale) || !Number.isFinite(Math.fround(scale)) || scale <= 0) {
    throw new Error('[PairedAttention] Explicit mask positions, causal flag and positive scale required.');
  }
  if (mask != null || (attnSoftcap !== null && attnSoftcap !== 0) || (options.bidirectionalSpanLength ?? 0) !== 0
    || (options.kvLayout != null && options.kvLayout !== 'contiguous') || options.kvPageTable != null
    || options.kvLenBuffer != null || options.indirectBuffer != null || options.outputGate != null
    || options.useFlashPrefill === true || options.useOrtFlashPrefill === true) {
    throw new Error('[PairedAttention] Declared contiguous causal/sliding attention required; masks, softcap, paging and fused epilogues need their own paired path.');
  }
  if (Q.dtype !== 'f32' || Q.precision !== 'two-f32' || Q.components !== 2 || Q.elementStrideBytes !== 8) {
    throw new Error('[PairedAttention] Explicit paired Q required.');
  }
  for (const value of [K, V]) {
    if (value.dtype !== 'f16' || value.precision !== undefined || value.components !== undefined || value.elementStrideBytes !== undefined) {
      throw new Error('[PairedAttention] Ordinary F16 K/V cache required.');
    }
  }
  const queryCount = seqLen * numHeads * headDim, cacheCount = kvLen * numKVHeads * headDim;
  /** @type {[import('../tensor.js').Tensor, number, number][]} */
  const storage = [[Q, queryCount, 8], [K, cacheCount, 2], [V, cacheCount, 2]];
  for (const [tensor, count, stride] of storage) {
    if (!Number.isSafeInteger(count) || count > 0xffffffff || tensor.buffer.size < count * stride
      || tensor.shape.reduce((product, value) => product * value, 1) < count) throw new Error('[PairedAttention] Logical tensor shape or storage is undersized.');
  }
  if (options.kernelPath != null) {
    const pinned = getKernelPathAttentionVariant(seqLen > 1 ? 'prefill' : 'decode', options.layerIdx, options.kernelPath);
    if (pinned !== 'paired') throw new Error('[PairedAttention] Execution path must explicitly select paired attention.');
  }
  const bytes = queryCount * 8;
  if (outputBuffer && (outputBuffer.size < bytes || [Q.buffer, K.buffer, V.buffer].includes(outputBuffer))) {
    throw new Error('[PairedAttention] Borrowed output must be distinct and sufficiently sized.');
  }
  const device = recorder?.device ?? getDevice();
  if (!device || numHeads > device.limits.maxComputeWorkgroupsPerDimension || seqLen > device.limits.maxComputeWorkgroupsPerDimension) {
    throw new Error('[PairedAttention] Device dispatch limits exceeded or no device.');
  }
  return withKernelOutput(recorder, outputBuffer, bytes, 'paired_attention_output', async output => {
    await unifiedKernelWrapper('attention', recorder, 'paired', [Q, K, V, output], {
      seq_len: seqLen, kv_len: kvLen, num_heads: numHeads, num_kv_heads: numKVHeads,
      head_dim: headDim, start_pos: startPos, kv_start: kvStart, sliding_window: slidingWindow,
      scale_hi: Math.fround(scale), scale_lo: Math.fround(scale - Math.fround(scale)), causal: causal ? 1 : 0,
    }, [numHeads, seqLen, 1]);
    return createTensor(output, 'f32', [seqLen, numHeads, headDim], 'paired_attention_output', PAIR);
  });
}
