import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { getDevice } from '../device.js';
import { unifiedKernelWrapper } from './kernel-execution.js';
import { getPipelineFast } from './pipeline-cache.js';
import { resolveGatedDeltaNormGateKernel } from './gated-delta-norm-gate.js';
import { getKernelConfig } from './kernel-configs.js';

/** @type {Readonly<import('../tensor.js').TensorRepresentation>} */
const REPRESENTATION = Object.freeze({ precision: 'two-f32', components: 2, elementStrideBytes: 8 });
const CONV_CONSTANTS = Object.freeze({ WORKGROUP_SIZE: 64 });
const GATE_CONSTANTS = Object.freeze({ WORKGROUP_SIZE: 64 });
const RECURRENT_CONSTANTS = Object.freeze({ WORKGROUP_SIZE: 128 });

/** The paired gate phase is selected by rule and must declare the model's activation.
 * @param {import('../../config/schema/inference.schema.js').LinearNormGateActivation} normGateActivation */
function resolvePairedGateVariant(normGateActivation) {
  return resolveGatedDeltaNormGateKernel('gated_delta_recurrent', 'gatedDeltaRecurrent', 'pairedGateVariant',
    { normGateActivation }).variant;
}

/** @type {typeof import('./paired-linear-attention.js').listPairedLinearAttentionPipelines} */
export function listPairedLinearAttentionPipelines(normGateActivation, replacements) {
  const gateVariant = resolvePairedGateVariant(normGateActivation);
  if (!replacements || typeof replacements !== 'object') {
    throw new Error('[PairedLinear] The resolved mechanism replacements of the declared closure are required.');
  }
  /** @type {import('./pipeline-cache.js').KernelPipelineRequest[]} */
  const phases = [
    { operation: 'gated_delta_conv', variant: 'paired', constants: CONV_CONSTANTS },
    { operation: 'qwen_linear_attention_prepare', variant: 'paired_beta', constants: GATE_CONSTANTS },
    { operation: 'qwen_linear_attention_prepare', variant: 'paired_decay_scale', constants: GATE_CONSTANTS },
    { operation: 'qwen_linear_attention_prepare', variant: 'paired_log_decay', constants: GATE_CONSTANTS },
    { operation: 'qwen_linear_attention_prepare', variant: 'paired_decay', constants: GATE_CONSTANTS },
    { operation: 'gated_delta_recurrent', variant: 'paired_qk', constants: RECURRENT_CONSTANTS },
    { operation: 'gated_delta_recurrent', variant: 'paired_state', constants: RECURRENT_CONSTANTS },
    { operation: 'gated_delta_recurrent', variant: 'paired_norm', constants: RECURRENT_CONSTANTS },
    { operation: 'gated_delta_recurrent', variant: gateVariant, constants: RECURRENT_CONSTANTS },
  ];
  // A replacement program carries its specialization constants as registry overrides.
  return phases.map(phase => {
    const replacement = replacements[`${phase.operation}/${phase.variant}`];
    return replacement === undefined ? phase : { operation: phase.operation, variant: replacement, constants: null };
  });
}

/**
 * Workgroup grid of the state phase. Programs declaring `headColumnBlocks` split each value
 * head into blocks of WORKGROUP_SIZE / K_SLICES columns and hold at most
 * K_SLICES * stateRowsPerLane state rows per column; the others run one workgroup per head.
 * @param {import('./pipeline-cache.js').KernelPipelineRequest} phase
 * @param {import('./linear-attention-core.js').LinearAttentionCoreLayerState} state
 * @returns {[number, number, number]}
 */
function resolveStateWorkgroups(phase, state) {
  const config = getKernelConfig(phase.operation, phase.variant);
  const geometry = config.variantMetadata?.dispatchGeometry;
  if (geometry === undefined) return [state.numVHeads, 1, 1];
  const lanes = Number(config.wgslOverrides?.WORKGROUP_SIZE), slices = Number(config.wgslOverrides?.K_SLICES);
  const rowsPerLane = Number(config.variantMetadata?.stateRowsPerLane);
  if (geometry !== 'headColumnBlocks' || !Number.isSafeInteger(lanes) || !Number.isSafeInteger(slices)
    || slices <= 0 || lanes % slices !== 0 || !Number.isSafeInteger(rowsPerLane) || state.headKDim > slices * rowsPerLane) {
    throw new Error(`[PairedLinear] ${phase.operation}/${phase.variant} cannot hold this state geometry.`);
  }
  return [state.numVHeads, Math.ceil(state.headVDim / (lanes / slices)), 1];
}

/** @type {typeof import('./paired-linear-attention.js').preparePairedLinearAttentionPipelines} */
export async function preparePairedLinearAttentionPipelines(device, signal, normGateActivation, replacements) {
  signal?.throwIfAborted();
  if (!device) throw new Error('[PairedLinear] Pipeline preparation requires the owning GPU device.');
  for (const { operation, variant, constants } of listPairedLinearAttentionPipelines(normGateActivation, replacements)) {
    await getPipelineFast(operation, variant, null, constants, device);
    signal?.throwIfAborted();
  }
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
  const [convPhase, betaPhase, decayScalePhase, logDecayPhase, decayPhase, qkPhase, statePhase, normPhase, gatePhase] =
    listPairedLinearAttentionPipelines(options.normGateActivation, /** @type {Readonly<Record<string, string>>} */ (options.mechanismReplacements));
  /** @param {import('./pipeline-cache.js').KernelPipelineRequest} phase
   * @param {Array<GPUBuffer | import('../tensor.js').Tensor>} bindings @param {number[]} workgroups
   * @param {Record<string, number>} uniform */
  const dispatch = (phase, bindings, uniform, workgroups) => unifiedKernelWrapper(phase.operation, recorder, phase.variant,
    bindings, uniform, workgroups, phase.constants);
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
  if (device.limits.maxComputeWorkgroupsPerDimension < 2
    || Math.ceil(numTokens * state.numVHeads / 64) > device.limits.maxComputeWorkgroupsPerDimension
    || Math.ceil(state.convDim / 64) > device.limits.maxComputeWorkgroupsPerDimension
    || Math.ceil(numTokens * state.valueDim / 128) > device.limits.maxComputeWorkgroupsPerDimension
    || state.numVHeads > device.limits.maxComputeWorkgroupsPerDimension
    || numTokens * Math.max(state.convDim, state.valueDim) * 8 > 0xffffffff) {
    throw new Error('[PairedLinear] Geometry exceeds dispatch or binding limits.');
  }
  const stateWorkgroups = resolveStateWorkgroups(statePhase, state);
  if (stateWorkgroups[1] > device.limits.maxComputeWorkgroupsPerDimension) throw new Error('[PairedLinear] Geometry exceeds dispatch or binding limits.');
  // Beta/decay prefix plus one paired logarithm input per token/head. Keeping
  // exp and log in separate gate phases bounds their native compilation.
  const gateBytes = numTokens * state.numVHeads * 24;
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
    await dispatch(convPhase, [qkv, state.convWeightGPU, state.convStateGPU, conv], uniform,
      [Math.ceil(state.convDim / 64), 1, 1]);
    await dispatch(betaPhase, [b, gates], uniform, [Math.ceil(numTokens * state.numVHeads / 64), 1, 1]);
    await dispatch(decayScalePhase, [state.aLogGPU, a, state.dtBiasGPU, gates], uniform,
      [Math.ceil(numTokens * state.numVHeads / 64), 2, 1]);
    await dispatch(logDecayPhase, [a, state.dtBiasGPU, gates], uniform, [Math.ceil(numTokens * state.numVHeads / 64), 1, 1]);
    await dispatch(decayPhase, [gates], uniform, [Math.ceil(numTokens * state.numVHeads / 64), 1, 1]);
    await dispatch(qkPhase, [conv], uniform, [state.numKHeads, 1, 1]);
    await dispatch(statePhase, [conv, gates, state.recurrentStateGPU, output], uniform, stateWorkgroups);
    await dispatch(normPhase, [state.normWeightGPU, output], uniform, [state.numVHeads, 1, 1]);
    await dispatch(gatePhase, [z, output], uniform, [Math.ceil(numTokens * state.valueDim / 128), 1, 1]);
    return createTensor(output, 'f32', [numTokens, state.valueDim], 'paired_linear_output', REPRESENTATION);
  } catch (error) { if (output) release(output); throw error; }
  finally { if (gates) release(gates); if (conv) release(conv); }
}
