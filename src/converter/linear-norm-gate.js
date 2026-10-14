import { LINEAR_NORM_GATE_ACTIVATIONS } from '../config/schema/manifest.schema.js';

// Source semantics of the linear-attention gated RMSNorm activation, as pinned
// in artifacts/qwen38-flash-next-20261013/source (Transformers e0eb2810):
// - qwen3_5: Qwen3_5RMSNormGated hardcodes activation = "silu". The config's
//   output_gate_type is not read; a value contradicting silu fails here.
// - qwen4_exp: Qwen4ExpTextRMSNormGated uses `output_gate_type or hidden_act`,
//   and validate_architecture rejects anything but "sigmoid" or "silu".
// In neither family does output_gate_type describe the full-attention output
// gate, which is always sigmoid.
const QWEN35_FAMILIES = new Set(['qwen3_5', 'qwen3_5_text']);
const QWEN4_EXP_FAMILIES = new Set(['qwen4_exp', 'qwen4_exp_text']);
/** @type {ReadonlySet<unknown>} */
const QWEN35_CONSISTENT_GATE_TYPES = new Set(['silu', 'swish']);
const LINEAR_LAYER_TYPES = new Set(['linear_attention', 'linear', 'gated_delta', 'gated_delta_net']);

/** @param {Record<string, unknown> | null | undefined} rawConfig @param {string} key @returns {unknown} */
function readSourceField(rawConfig, key) {
  if (!rawConfig || typeof rawConfig !== 'object') return undefined;
  const topLevel = rawConfig[key];
  if (topLevel !== undefined && topLevel !== null) return topLevel;
  const textConfig = /** @type {Record<string, unknown> | null | undefined} */ (rawConfig.text_config);
  if (textConfig && typeof textConfig === 'object' && textConfig[key] !== undefined) return textConfig[key];
  return topLevel;
}

/** @type {readonly unknown[]} */
const ACTIVATIONS = LINEAR_NORM_GATE_ACTIVATIONS;

/** @type {typeof import('./linear-norm-gate.js').normalizeSourceModelFamily} */
export function normalizeSourceModelFamily(rawConfig) {
  return String(readSourceField(rawConfig, 'model_type') ?? '').trim().toLowerCase();
}

/** @type {typeof import('./linear-norm-gate.js').sourceDeclaresLinearAttention} */
export function sourceDeclaresLinearAttention(rawConfig) {
  const layerTypes = readSourceField(rawConfig, 'layer_types');
  return Array.isArray(layerTypes)
    && layerTypes.some(entry => LINEAR_LAYER_TYPES.has(String(entry ?? '').trim().toLowerCase()));
}

/** @type {typeof import('./linear-norm-gate.js').resolveSourceLinearNormGateActivation} */
export function resolveSourceLinearNormGateActivation(rawConfig) {
  const family = normalizeSourceModelFamily(rawConfig);
  const gateType = readSourceField(rawConfig, 'output_gate_type');
  if (QWEN4_EXP_FAMILIES.has(family)) {
    // Python `output_gate_type or hidden_act`: None and "" fall back.
    const activation = (typeof gateType === 'string' && gateType !== '') ? gateType : readSourceField(rawConfig, 'hidden_act');
    if (activation !== 'sigmoid' && activation !== 'silu') {
      throw new Error(
        `Source model_type "${family}" resolves the linear-attention norm gate to ${JSON.stringify(activation)} ` +
        '(output_gate_type or hidden_act); upstream qwen4_exp accepts only "sigmoid" or "silu".'
      );
    }
    return activation;
  }
  if (QWEN35_FAMILIES.has(family)) {
    if (gateType !== undefined && gateType !== null && !QWEN35_CONSISTENT_GATE_TYPES.has(gateType)) {
      throw new Error(
        `Source model_type "${family}" declares output_gate_type=${JSON.stringify(gateType)}, but upstream ` +
        'Qwen3_5RMSNormGated hardcodes silu and does not read it. Resolve the source contradiction; ' +
        'Doppler will not honor a gate the reference implementation ignores.'
      );
    }
    return 'silu';
  }
  return null;
}

/**
 * Conversion contract: an authored inference.normalization.linearNormGateActivation
 * must equal the source semantics, and a source whose activation differs from
 * the legacy default must author it. Pre-field Qwen3.5 configs stay valid.
 * @type {typeof import('./linear-norm-gate.js').assertConvertedLinearNormGateActivation}
 */
export function assertConvertedLinearNormGateActivation(inference, rawConfig, modelId) {
  const declared = inference?.normalization?.linearNormGateActivation;
  if (declared !== undefined && !ACTIVATIONS.includes(declared)) {
    throw new Error(
      `Config for "${modelId}" has inference.normalization.linearNormGateActivation=${JSON.stringify(declared)}; ` +
      `supported: ${LINEAR_NORM_GATE_ACTIVATIONS.join(', ')}.`
    );
  }
  const source = resolveSourceLinearNormGateActivation(rawConfig);
  if (source === null) {
    if (declared === undefined && sourceDeclaresLinearAttention(rawConfig)) {
      throw new Error(
        `Config for "${modelId}" converts linear-attention source model_type ` +
        `"${normalizeSourceModelFamily(rawConfig)}" without a known norm-gate mapping; author ` +
        'inference.normalization.linearNormGateActivation explicitly.'
      );
    }
    return;
  }
  if (declared === undefined) {
    if (source !== 'silu') {
      throw new Error(
        `Config for "${modelId}" must author inference.normalization.linearNormGateActivation="${source}"; ` +
        'an absent field means silu.'
      );
    }
    return;
  }
  if (declared !== source) {
    throw new Error(
      `Config for "${modelId}" declares inference.normalization.linearNormGateActivation="${declared}", ` +
      `but the source model_type "${normalizeSourceModelFamily(rawConfig)}" applies ${source}.`
    );
  }
}
