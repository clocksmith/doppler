import type { LinearNormGateActivation } from '../../config/schema/inference.schema.js';
import type { KernelConfig } from './kernel-configs.js';

/** Return the activation unchanged, or throw when it is not a supported gated-RMSNorm activation. */
export declare function requireNormGateActivation(value: unknown, label: string): LinearNormGateActivation;

/**
 * Throw unless the kernel's registry `variantMetadata.normGateActivation`
 * equals the model-declared activation (lane-mismatch guard).
 */
export declare function assertKernelNormGateActivation(
  config: KernelConfig,
  normGateActivation: LinearNormGateActivation
): KernelConfig;

/** Select a gated-delta gate kernel via `kernels.<ruleGroup>.<ruleName>` and guard its declared activation. */
export declare function resolveGatedDeltaNormGateKernel(
  operation: 'gated_delta_recurrent' | 'gated_delta_fused_decode',
  ruleGroup: 'gatedDeltaRecurrent' | 'gatedDeltaFusedDecode',
  ruleName: 'scalarVariant' | 'pairedGateVariant',
  context: { normGateActivation: LinearNormGateActivation; inputDtype?: 'f16' | 'f32'; stateComponents?: number }
): KernelConfig;
