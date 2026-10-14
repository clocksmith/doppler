import { selectRuleValue } from '../../rules/rule-registry.js';
import { LINEAR_NORM_GATE_ACTIVATIONS } from '../../config/manifest-inference-validation.js';
import { getKernelConfig } from './kernel-configs.js';

/** @type {readonly unknown[]} */
const NORM_GATE_ACTIVATIONS = LINEAR_NORM_GATE_ACTIVATIONS;

/** @type {typeof import('./gated-delta-norm-gate.js').requireNormGateActivation} */
export function requireNormGateActivation(value, label) {
  if (!NORM_GATE_ACTIVATIONS.includes(value)) {
    throw new Error(
      `${label} requires the resolved linear-attention norm-gate activation ` +
      `(${NORM_GATE_ACTIVATIONS.join(' | ')}); got ${JSON.stringify(value)}.`
    );
  }
  return /** @type {import('../../config/schema/inference.schema.js').LinearNormGateActivation} */ (value);
}

/**
 * Lane-mismatch guard: the registry metadata of the kernel that will apply
 * the gated-RMSNorm gate must declare the same activation as the model.
 * @type {typeof import('./gated-delta-norm-gate.js').assertKernelNormGateActivation}
 */
export function assertKernelNormGateActivation(config, normGateActivation) {
  const declared = config?.variantMetadata?.normGateActivation;
  if (declared !== normGateActivation) {
    throw new Error(
      `Kernel ${config?.operation}/${config?.variant} (${config?.shaderFile}) declares norm-gate activation ` +
      `${JSON.stringify(declared ?? null)}, but the model declares "${normGateActivation}". ` +
      'A gated-delta kernel cannot apply a different gate than the manifest.'
    );
  }
  return config;
}

/**
 * Select a gate-applying gated-delta kernel through its rule map, then guard
 * its registry-declared activation.
 * @type {typeof import('./gated-delta-norm-gate.js').resolveGatedDeltaNormGateKernel}
 */
export function resolveGatedDeltaNormGateKernel(operation, ruleGroup, ruleName, context) {
  requireNormGateActivation(context?.normGateActivation, `${operation} kernel selection`);
  const variant = selectRuleValue('kernels', ruleGroup, ruleName, context);
  return assertKernelNormGateActivation(getKernelConfig(operation, variant), context.normGateActivation);
}
