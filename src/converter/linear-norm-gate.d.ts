import type { LinearNormGateActivation } from '../config/schema/inference.schema.js';

/** Lowercased source `model_type`, top-level first, then `text_config`. */
export declare function normalizeSourceModelFamily(rawConfig: Record<string, unknown> | null | undefined): string;

/** Whether the source config's `layer_types` include a linear-attention layer. */
export declare function sourceDeclaresLinearAttention(rawConfig: Record<string, unknown> | null | undefined): boolean;

/**
 * Source-family gated-RMSNorm activation: `silu` for qwen3_5 (hardcoded
 * upstream; a contradicting output_gate_type throws), `output_gate_type or
 * hidden_act` for qwen4_exp (only sigmoid/silu accepted), `null` for families
 * without a known mapping.
 */
export declare function resolveSourceLinearNormGateActivation(
  rawConfig: Record<string, unknown> | null | undefined
): LinearNormGateActivation | null;

/** Throw when an authored activation disagrees with the source, or a non-legacy source omits it. */
export declare function assertConvertedLinearNormGateActivation(
  inference: { normalization?: { linearNormGateActivation?: unknown } } | null | undefined,
  rawConfig: Record<string, unknown> | null | undefined,
  modelId: string
): void;
