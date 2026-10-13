import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { getDevice } from '../device.js';
import { unifiedKernelWrapper } from './kernel-execution.js';
import { getPipelineFast } from './pipeline-cache.js';

/** @type {Readonly<import('../tensor.js').TensorRepresentation>} */
const REPRESENTATION = Object.freeze({ precision: 'two-f32', components: 2, elementStrideBytes: 8 });
const CONV_CONSTANTS = Object.freeze({ WORKGROUP_SIZE: 64 });
const GATE_CONSTANTS = Object.freeze({ WORKGROUP_SIZE: 64 });
const RECURRENT_CONSTANTS = Object.freeze({ WORKGROUP_SIZE: 128 });

/** @type {typeof import('./paired-linear-attention.js').preparePairedLinearAttentionPipelines} */
export async function preparePairedLinearAttentionPipelines(device, signal) {
  signal?.throwIfAborted();
  if (!device) throw new Error('[PairedLinear] Pipeline preparation requires the owning GPU device.');
  await getPipelineFast('gated_delta_conv', 'paired', null, CONV_CONSTANTS, device);
  signal?.throwIfAborted();
  await getPipelineFast('qwen_linear_attention_prepare', 'paired_beta', null, GATE_CONSTANTS, device);
  signal?.throwIfAborted();
  await getPipelineFast('qwen_linear_attention_prepare', 'paired_decay_scale', null, GATE_CONSTANTS, device);
  signal?.throwIfAborted();
  await getPipelineFast('qwen_linear_attention_prepare', 'paired_log_decay', null, GATE_CONSTANTS, device);
  signal?.throwIfAborted();
  await getPipelineFast('qwen_linear_attention_prepare', 'paired_decay', null, GATE_CONSTANTS, device);
  signal?.throwIfAborted();
  await getPipelineFast('gated_delta_recurrent', 'paired_qk', null, RECURRENT_CONSTANTS, device);
  signal?.throwIfAborted();
  await getPipelineFast('gated_delta_recurrent', 'paired_state', null, RECURRENT_CONSTANTS, device);
  signal?.throwIfAborted();
  await getPipelineFast('gated_delta_recurrent', 'paired_norm', null, RECURRENT_CONSTANTS, device);
  signal?.throwIfAborted();
  await getPipelineFast('gated_delta_recurrent', 'paired_gate', null, RECURRENT_CONSTANTS, device);
  signal?.throwIfAborted();
}

/** @param {import('../tensor.js').Tensor} tensor @param {number} count */
function assertPair(tensor, count) {
  if (!Number.isSafeInteger(count * 8) || count * 8 > 0xffffffff || tensor.dtype !== 'f32' || tensor.precision !== 'two-f32' || tensor.components !== 2
    || tensor.elementStrideBytes !== 8 || tensor.buffer.size < count * 8) {
    throw new Error('[PairedLinear] Every projected input must carry an adequately sized explicit paired representation.');
  }
}

/** @type {typeof import('./paired-linear-attention.js').executePairedLinearAttention} */
export async function executePairedLinearAttention(qkv, z, a, b, state, options) {
  const { numTokens, recorder = null } = options;
  const dimensions = [numTokens, state.convDim, state.convKernelSize, state.numVHeads, state.numKHeads,
    state.headKDim, state.headVDim, state.qSize, state.kSize, state.valueDim, state.qRep];
  if (!dimensions.every(v => Number.isSafeInteger(v) && v > 0 && v <= 0xffffff)
    || state.headKDim > 128 || state.headVDim > 128 || state.numVHeads !== state.numKHeads * state.qRep
    || state.qSize !== state.numKHeads * state.headKDim || state.kSize !== state.qSize
    || state.valueDim !== state.numVHeads * state.headVDim || state.convDim !== state.qSize + state.kSize + state.valueDim) {
    throw new Error('[PairedLinear] Explicit consistent geometry with head dimensions at most128 is required.');
  }
  if (options.outputDtype !== 'f32' || state.convStatePrecision !== 'two-f32' || state.convStateComponents !== 2
    || state.recurrentStatePrecision !== 'two-f32' || state.recurrentStateComponents !== 2) {
    throw new Error('[PairedLinear] Paired convolution/recurrent state and F32 component output are required.');
  }
  if (state.normMode !== 'shared' && state.normMode !== 'per_head') throw new Error('[PairedLinear] Explicit norm mode required.');
  if (typeof state.normWeightOffset !== 'boolean') throw new Error('[PairedLinear] Explicit norm weight offset required.');
  const qeps = options.qkL2NormEps, eps = state.rmsNormEps;
  if (typeof qeps !== 'number' || ![qeps, eps].every(v => typeof v === 'number' && Number.isFinite(v) && v > 0 && Math.fround(v) > 0)) {
    throw new Error('[PairedLinear] Positive source epsilons required.');
  }
  const abPacked = options.abPacked === true, qkvzPacked = options.qkvzPacked === true;
  const bOffset = abPacked ? options.bProjOffsetElements : 0;
  if (!Number.isSafeInteger(bOffset) || bOffset === undefined || bOffset < 0) throw new Error('[PairedLinear] Explicit valid packed B offset required.');
  assertPair(qkv, numTokens * (state.convDim + (qkvzPacked ? state.valueDim : 0)));
  assertPair(z, numTokens * (qkvzPacked ? state.convDim + state.valueDim : state.valueDim));
  assertPair(a, numTokens * state.numVHeads); assertPair(b, bOffset + numTokens * state.numVHeads);
  const buffers = [
    [state.convStateGPU, state.convDim * state.convKernelSize * 8],
    [state.recurrentStateGPU, state.numVHeads * state.headKDim * state.headVDim * 8],
    [state.convWeightGPU, state.convDim * state.convKernelSize * 4],
    [state.dtBiasGPU, state.numVHeads * 4], [state.aLogGPU, state.numVHeads * 4],
    [state.normWeightGPU, (state.normMode === 'shared' ? state.headVDim : state.valueDim) * 4],
  ];
  for (const [buffer, bytes] of /** @type {Array<[GPUBuffer, number]>} */ (buffers)) {
    if (!Number.isSafeInteger(bytes) || bytes > 0xffffffff || !buffer || buffer.size < bytes) throw new Error('[PairedLinear] Invalid persistent binding size.');
  }
  const device = recorder?.device ?? getDevice();
  if (!device || device.limits.maxStorageBuffersPerShaderStage < 9) throw new Error('[PairedLinear] Nine storage bindings per stage are required.');
  if (Math.ceil(numTokens * state.numVHeads / 64) > device.limits.maxComputeWorkgroupsPerDimension
    || Math.ceil(state.convDim / 64) > device.limits.maxComputeWorkgroupsPerDimension
    || Math.ceil(numTokens * state.valueDim / 128) > device.limits.maxComputeWorkgroupsPerDimension
    || state.numVHeads > device.limits.maxComputeWorkgroupsPerDimension
    || numTokens * Math.max(state.convDim, state.valueDim) * 8 > 0xffffffff) {
    throw new Error('[PairedLinear] Geometry exceeds dispatch or binding limits.');
  }
  const gateBytes = numTokens * state.numVHeads * 16;
  if (!Number.isSafeInteger(gateBytes) || gateBytes > 0xffffffff
    || gateBytes > device.limits.maxStorageBufferBindingSize || gateBytes > device.limits.maxBufferSize) {
    throw new Error('[PairedLinear] Prepared gate buffer exceeds storage or allocation limits.');
  }
  const reads = [qkv.buffer, z.buffer, a.buffer, b.buffer, state.convWeightGPU, state.dtBiasGPU, state.aLogGPU, state.normWeightGPU];
  if (state.convStateGPU === state.recurrentStateGPU || reads.includes(state.convStateGPU) || reads.includes(state.recurrentStateGPU)) {
    throw new Error('[PairedLinear] Persistent writable state must not alias inputs or each other.');
  }
  const uniform = { num_tokens: numTokens, conv_dim: state.convDim, conv_kernel_size: state.convKernelSize,
    num_v_heads: state.numVHeads, num_k_heads: state.numKHeads, head_k_dim: state.headKDim, head_v_dim: state.headVDim,
    q_size: state.qSize, k_size: state.kSize, value_dim: state.valueDim, q_rep: state.qRep,
    norm_mode: Number(state.normMode === 'per_head'), rms_eps_hi: Math.fround(eps), rms_eps_lo: eps - Math.fround(eps),
    qk_eps_hi: Math.fround(qeps), qk_eps_lo: qeps - Math.fround(qeps), packed_flags: Number(abPacked) | (Number(qkvzPacked) << 1),
    b_offset: bOffset, norm_weight_offset: Number(state.normWeightOffset) };
  let conv = null, gates = null, output = null;
  /** @param {GPUBuffer} buffer */
  const release = buffer => { if (recorder) recorder.trackTemporaryBuffer(buffer); else releaseBuffer(buffer); };
  try {
    conv = acquireBuffer(numTokens * state.convDim * 8, undefined, 'paired_linear_conv');
    gates = acquireBuffer(gateBytes, undefined, 'paired_linear_gates');
    output = acquireBuffer(numTokens * state.valueDim * 8, undefined, 'paired_linear_output');
    await unifiedKernelWrapper('gated_delta_conv', recorder, 'paired', [qkv, state.convWeightGPU, state.convStateGPU, conv], uniform,
      [Math.ceil(state.convDim / 64), 1, 1], CONV_CONSTANTS);
    await unifiedKernelWrapper('qwen_linear_attention_prepare', recorder, 'paired_beta',
      [b, gates], uniform,
      [Math.ceil(numTokens * state.numVHeads / 64), 1, 1], GATE_CONSTANTS);
    await unifiedKernelWrapper('qwen_linear_attention_prepare', recorder, 'paired_decay_scale',
      [state.aLogGPU, gates], uniform,
      [Math.ceil(numTokens * state.numVHeads / 64), 1, 1], GATE_CONSTANTS);
    await unifiedKernelWrapper('qwen_linear_attention_prepare', recorder, 'paired_log_decay',
      [a, state.dtBiasGPU, gates], uniform,
      [Math.ceil(numTokens * state.numVHeads / 64), 1, 1], GATE_CONSTANTS);
    await unifiedKernelWrapper('qwen_linear_attention_prepare', recorder, 'paired_decay',
      [gates], uniform,
      [Math.ceil(numTokens * state.numVHeads / 64), 1, 1], GATE_CONSTANTS);
    await unifiedKernelWrapper('gated_delta_recurrent', recorder, 'paired_qk',
      [conv], uniform, [state.numKHeads, 1, 1], RECURRENT_CONSTANTS);
    await unifiedKernelWrapper('gated_delta_recurrent', recorder, 'paired_state',
      [conv, gates, state.recurrentStateGPU, output], uniform,
      [state.numVHeads, 1, 1], RECURRENT_CONSTANTS);
    await unifiedKernelWrapper('gated_delta_recurrent', recorder, 'paired_norm',
      [state.normWeightGPU, output], uniform,
      [state.numVHeads, 1, 1], RECURRENT_CONSTANTS);
    await unifiedKernelWrapper('gated_delta_recurrent', recorder, 'paired_gate',
      [z, output], uniform, [Math.ceil(numTokens * state.valueDim / 128), 1, 1], RECURRENT_CONSTANTS);
    return createTensor(output, 'f32', [numTokens, state.valueDim], 'paired_linear_output', REPRESENTATION);
  } catch (error) { if (output) release(output); throw error; }
  finally { if (gates) release(gates); if (conv) release(conv); }
}
