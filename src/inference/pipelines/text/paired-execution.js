import { getLayerSteps, getKernelPathStepConfig } from '../../../config/kernel-path-loader.js';
import {
  assertLinearNormGateClosure,
  assertPairedExecutionMechanismClosure,
} from '../../../config/execution-registry-contract.js';
import { resolveLinearNormGateActivation } from '../../../config/schema/manifest.schema.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';

/**
 * Paired closure for the manifest's declared norm-gate activation. Used by Run
 * admission and Rig signing; pre-field manifests resolve to the silu closure.
 * @type {typeof import('./paired-execution.js').assertPairedLinearMechanismClosure}
 */
export function assertPairedLinearMechanismClosure(manifestInference) {
  const graph = manifestInference?.execution;
  if (!graph) throw new Error('[PairedExecution] An explicit execution graph is required.');
  const normGateActivation = resolveLinearNormGateActivation(manifestInference.normalization,
    'manifest.inference.normalization');
  assertLinearNormGateClosure(graph, normGateActivation);
  const context = { normGateActivation };
  /** @type {string | null} */
  const anchor = selectRuleValue('kernels', 'gatedDeltaRecurrent', 'pairedClosureAnchorVariant', context);
  /** @type {string} */
  const gate = selectRuleValue('kernels', 'gatedDeltaRecurrent', 'pairedGateVariant', context);
  /** @type {Array<readonly [string, string]>} */
  const mechanisms = anchor === null ? [] : [['gated_delta_recurrent', anchor]];
  mechanisms.push(['gated_delta_recurrent', gate]);
  assertPairedExecutionMechanismClosure(graph, mechanisms);
}

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
  const metadata = config?.variantMetadata;
  if (!config || (variant != null && config.variant !== variant)
    || ((operation === 'matmul' || operation === 'attention') && (metadata?.inputPrecision !== 'two-f32'
      || metadata?.outputPrecision !== outputPrecision))
    || (operation === 'attention' && metadata?.kvDtype !== 'f16')) {
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
  assertPairedLinearMechanismClosure(manifest.inference);
  for (let layer = 0; layer < count; layer++) for (const phase of /** @type {const} */ (['prefill', 'decode'])) {
    const steps = getLayerSteps(path, layer, phase);
    for (const role of ['input_norm', 'post_attn_norm']) requireStep(steps, role, 'rmsnorm', 'paired');
    for (const role of ['gate_proj', 'up_proj', 'down_proj', 'o_proj']) requireStep(steps, role, 'matmul', null);
    requireStep(steps, 'activation', 'paired_pointwise', 'silu');
    for (const role of ['attn_residual', 'ffn_residual']) requireStep(steps, role, 'paired_pointwise', 'add');
    if (types[layer] === 'full_attention') {
      for (const role of ['q_proj', 'k_proj', 'v_proj']) requireStep(steps, role, 'matmul', null);
      for (const role of ['rope_q', 'rope_k']) requireStep(steps, role, 'rope_qk', 'paired');
      // Any registry attention program declaring paired Q/output over F16 K/V.
      requireStep(steps, 'attention', 'attention', null);
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
