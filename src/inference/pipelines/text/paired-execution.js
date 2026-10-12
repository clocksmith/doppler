import { getLayerSteps, getKernelPathStepConfig } from '../../../config/kernel-path-loader.js';
import { getKernelConfigs } from '../../../config/kernel-registry-contract.js';

/**
 * @param {import('../../../config/schema/kernel-path.schema.js').KernelStepSchema[]} steps
 * @param {string} role
 * @param {string} operation
 * @param {string | null} variant
 * @param {string} outputPrecision
 */
function requireStep(steps, role, operation, variant, outputPrecision = 'two-f32') {
  const matches = steps.filter(step => step.op === role);
  if (matches.length !== 1) throw new Error(`[PairedExecution] Exactly one ${role} step is required.`);
  const step = matches[0];
  const config = getKernelPathStepConfig(operation, step);
  if (!config || (variant != null && config.variant !== variant)
    || (operation === 'matmul' && (config.variantMetadata?.inputPrecision !== 'two-f32'
      || config.variantMetadata?.outputPrecision !== outputPrecision))) {
    throw new Error(`[PairedExecution] ${role} is not bound to the declared paired ${operation} implementation.`);
  }
  if ([step.precision?.activationDtype, step.precision?.inputDtype, step.precision?.outputDtype]
    .some(dtype => dtype != null && dtype !== 'f32')) {
    throw new Error(`[PairedExecution] ${role} cannot narrow paired components.`);
  }
  return config;
}

// Admission checks the actual bound graph, not only the requested session flag.
/** @type {typeof import('./paired-execution.js').validatePairedExecution} */
export function validatePairedExecution(path, session, manifest) {
  if (!path || path.activationDtype !== 'f32' || session.linearRecurrentStatePrecision !== 'two-f32'
    || session.kvcache?.kvDtype !== 'f16' || session.kvcache?.layout !== 'contiguous') {
    throw new Error('[PairedExecution] A complete bound paired graph, two-F32 recurrent state and contiguous F16 KV are required.');
  }
  const types = manifest?.inference?.layerPattern?.layerTypes;
  const architecture = manifest?.architecture;
  const count = architecture && typeof architecture === 'object' ? architecture.numLayers : undefined;
  if (!Array.isArray(types) || typeof count !== 'number' || !Number.isSafeInteger(count) || count < 1 || types.length !== count
    || types.some(type => type !== 'linear_attention' && type !== 'full_attention')) {
    throw new Error('[PairedExecution] Explicit complete linear/full-attention layer geometry is required.');
  }
  const graph = manifest.inference.execution;
  if (!graph) throw new Error('[PairedExecution] An explicit execution graph is required.');
  const mechanisms = (graph.mechanismKernels ?? []).map(key => graph.kernels[key]);
  const registry = getKernelConfigs();
  const requiredMechanisms = [
    ['cast', 'f32_to_two_f32'], ['cast', 'two_f32_to_f16'],
    ['gated_delta_conv', 'paired'], ['gated_delta_recurrent', 'paired'],
    ['split_qg', 'paired'], ['split_qkv', 'paired'], ['rope_precompute', 'paired'],
    ['paired_pointwise', 'sigmoid'], ['paired_pointwise', 'scale'],
  ];
  for (const [operation, variant] of requiredMechanisms) {
    const expected = registry[operation]?.[variant];
    if (!expected || !mechanisms.some(step => step && getKernelPathStepConfig(operation, step)?.variant === variant
      && typeof step.digest === 'string' && /^sha256:[a-f0-9]{64}$/.test(step.digest))) {
      throw new Error(`[PairedExecution] Missing pinned mechanism ${operation}/${variant}.`);
    }
  }
  for (let layer = 0; layer < count; layer++) for (const phase of /** @type {const} */ (['prefill', 'decode'])) {
    const steps = getLayerSteps(path, layer, phase);
    for (const role of ['input_norm', 'post_attn_norm']) requireStep(steps, role, 'rmsnorm', 'paired');
    for (const role of ['gate_proj', 'up_proj', 'down_proj', 'o_proj']) requireStep(steps, role, 'matmul', null);
    requireStep(steps, 'activation', 'paired_pointwise', 'silu');
    for (const role of ['attn_residual', 'ffn_residual']) requireStep(steps, role, 'paired_pointwise', 'add');
    if (types[layer] === 'full_attention') {
      for (const role of ['q_proj', 'k_proj', 'v_proj']) requireStep(steps, role, 'matmul', null);
      for (const role of ['rope_q', 'rope_k']) requireStep(steps, role, 'rope_qk', 'paired');
      requireStep(steps, 'attention', 'attention', 'paired');
    } else {
      for (const role of ['linear_qkv_proj', 'linear_z_proj', 'linear_a_proj', 'linear_b_proj']) {
        requireStep(steps, role, 'matmul', null);
      }
    }
    if (steps.some(step => step.op === 'cast' || step.op === 'ffn' || step.op === 'ffn_gate_up')) {
      throw new Error('[PairedExecution] Scalar casts or fused FFN cannot occur in the paired graph.');
    }
  }
  const post = path.postLayer ?? [];
  requireStep(post, 'final_norm', 'rmsnorm', 'paired');
  for (const role of ['lm_head', 'lm_head_prefill']) requireStep(post, role, 'matmul', null, 'f32');
  return { activationPrecision: 'two-f32', activationComponents: 2, activationElementStrideBytes: 8 };
}
