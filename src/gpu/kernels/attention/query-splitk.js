import { getDevice } from '../../device.js';
import { acquireBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';
import { createTensor } from '../../tensor.js';
import { getKernelConfig } from '../kernel-configs.js';
import { unifiedKernelWrapper, withKernelOutput } from '../kernel-execution.js';
import { getKvLenFallbackBuffer, getPageTableFallbackBuffer } from './executors-components.js';

/** @type {typeof import('./query-splitk.js').executeQuerySplitK} */
export async function executeQuerySplitK(recorder, variant, Q, K, V, numHeads, headDim, options) {
  const device = recorder?.device ?? getDevice();
  if (!device) throw new Error('[Query split-K] An initialized device is required.');
  const { seqLen, kvLen, numKVHeads, scale, startPos } = options;
  if (![seqLen, kvLen, numHeads, numKVHeads, headDim, startPos + 1].every(n => Number.isSafeInteger(n) && n > 0)
    || headDim !== 256 || numHeads % numKVHeads !== 0 || !Number.isFinite(scale) || scale <= 0
    || !options.causal || startPos + seqLen !== kvLen || options.attnSoftcap !== 0 || (options.slidingWindow !== 0 && options.slidingWindow !== null)
    || options.kvStart !== 0 || (options.kvLayout !== null && options.kvLayout !== 'contiguous')
    || options.bidirectionalSpanLength !== 0) {
    throw new Error('[Query split-K] Requires explicit head_dim 256, integral GQA and contiguous unwindowed causal F32 attention: '
      + JSON.stringify({ seqLen, kvLen, numHeads, numKVHeads, headDim, scale, startPos, causal: options.causal,
        softcap: options.attnSoftcap, window: options.slidingWindow, kvStart: options.kvStart,
        layout: options.kvLayout, spanLength: options.bidirectionalSpanLength }));
  }
  const config = getKernelConfig('attention', variant);
  const tile = config.wgslOverrides?.QUERY_TILE, splits = config.wgslOverrides?.KV_SPLITS;
  const reducer = config.variantMetadata?.reduceVariant;
  if (config.variantMetadata?.querySplitK !== true || typeof reducer !== 'string'
    || typeof tile !== 'number' || !Number.isSafeInteger(tile) || tile < 1
    || typeof splits !== 'number' || !Number.isSafeInteger(splits) || splits < 1) {
    throw new Error('[Query split-K] Registered producer geometry and reducer identity are required.');
  }
  const reduce = getKernelConfig('attention', reducer);
  if (reduce.wgslOverrides?.BLOCK_SIZE !== tile || reduce.wgslOverrides?.MAX_KV_SPLITS !== splits) {
    throw new Error('[Query split-K] Producer and reducer geometry differ.');
  }
  const groups = Math.ceil(seqLen / tile) * numHeads, statsCount = groups * splits * tile;
  const outputBytes = seqLen * numHeads * headDim * 4;
  const temporaryBytes = [statsCount * headDim * 4, statsCount * 4, statsCount * 4];
  if ([groups, splits].some(n => n > device.limits.maxComputeWorkgroupsPerDimension)
    || [outputBytes, ...temporaryBytes].some(n => !Number.isSafeInteger(n) || n < 4
      || n > Math.min(0xffffffff, device.limits.maxStorageBufferBindingSize))) {
    throw new Error('[Query split-K] Declared dispatch or storage exceeds device limits.');
  }
  /** @type {Array<[import('../../tensor.js').Tensor, number]>} */
  const inputs = [[Q, outputBytes], [K, kvLen * numKVHeads * headDim * 4], [V, kvLen * numKVHeads * headDim * 4]];
  for (const [tensor, bytes] of inputs) {
    if (tensor.dtype !== 'f32' || tensor.precision != null || tensor.buffer.size < bytes
      || !(tensor.buffer.usage & GPUBufferUsage.STORAGE)) {
      throw new Error('[Query split-K] Scalar F32 inputs must cover the declared geometry.');
    }
  }
  if (options.outputBuffer && [Q.buffer, K.buffer, V.buffer].includes(options.outputBuffer)) {
    throw new Error('[Query split-K] Output must not alias an input.');
  }
  return withKernelOutput(recorder ?? device, options.outputBuffer, outputBytes, 'query_splitk_output', async output => {
    /** @type {GPUBuffer[]} */
    const owned = [];
    try {
      for (const bytes of temporaryBytes) {
        const buffer = recorder
          ? recorder.createTempBuffer(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 'query_splitk_partial')
          : acquireBuffer(bytes, undefined, 'query_splitk_partial');
        owned.push(buffer);
      }
      const [partial, maximum, denominator] = owned;
      const uniforms = { num_heads: numHeads, num_kv_heads: numKVHeads, head_dim: headDim,
        seq_len: kvLen, query_len: seqLen, scale, is_causal: 1, start_pos: startPos,
        attn_softcap: 0, sliding_window: 0, kv_len_source: options.kvLenBuffer ? 1 : 0,
        kv_start: 0, page_size: options.kvPageSize, kv_layout: 0,
        bidirectional_span_start: 0, bidirectional_span_length: 0 };
      await unifiedKernelWrapper('attention', recorder ?? device, variant,
        [Q.buffer, K.buffer, V.buffer, partial, options.kvLenBuffer ?? getKvLenFallbackBuffer(device),
          options.kvPageTable ?? getPageTableFallbackBuffer(device), maximum, denominator],
        uniforms, [groups, splits, 1], null, null, 'attention_splitk');
      await unifiedKernelWrapper('attention', recorder ?? device, reducer, [partial, maximum, denominator, output],
        { num_heads: numHeads, query_len: seqLen, num_kv_splits: splits },
        [Math.ceil(seqLen * numHeads / reduce.workgroupSize[0]), headDim / 4, 1], null, null, 'attention_splitk_reduce');
      return createTensor(output, 'f32', [seqLen, numHeads, headDim], 'attention_output');
    } finally { if (!recorder) for (const buffer of owned) releaseBuffer(buffer); }
  });
}
