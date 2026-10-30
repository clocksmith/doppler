import { DEFAULT_LINEAR_ATTENTION_SOURCE_LAYOUT } from '../../../config/schema/manifest.schema.js';
import { resolveDeclaredMechanismReplacements } from '../../../config/execution-registry-contract.js';

/** @param {unknown} value */
function parsePositiveInt(value) {
  const num = Number(value);
  if (!Number.isFinite(num) || num <= 0) return null;
  return Math.trunc(num);
}

/** @param {unknown} value @param {unknown} sharedFlag @param {string} modelId
 * @returns {import('../../../config/schema/index.js').LinearNormMode | null} */
function parseLinearNormMode(value, sharedFlag = null, modelId = 'unknown') {
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (normalized === 'shared') return 'shared';
    if (normalized === 'per_head' || normalized === 'per-head' || normalized === 'perhead') {
      return 'per_head';
    }
    throw new Error(
      `Manifest "${modelId}" has unsupported linear_norm_mode="${value}". ` +
      'Supported values: "shared", "per_head".'
    );
  }
  if (typeof sharedFlag === 'boolean') {
    return sharedFlag ? 'shared' : 'per_head';
  }
  return null;
}

/** @type {typeof import('./config-components.js').resolveLinearAttentionConfig} */
export function resolveLinearAttentionConfig(arch, config, inf, manifest, modelId) {
  const layout=arch.linearAttentionSourceLayout ?? config.linear_attention_source_layout ?? DEFAULT_LINEAR_ATTENTION_SOURCE_LAYOUT;
  if (layout !== 'hf' && layout !== 'gguf-v1') throw new Error(`Unsupported linear-attention source layout: ${layout}`);
  return {
    linearAttentionSourceLayout: layout,
    linearNumKeyHeads: parsePositiveInt(arch.linearNumKeyHeads ?? config.linear_num_key_heads),
    linearNumValueHeads: parsePositiveInt(arch.linearNumValueHeads ?? config.linear_num_value_heads),
    linearKeyHeadDim: parsePositiveInt(arch.linearKeyHeadDim ?? config.linear_key_head_dim),
    linearValueHeadDim: parsePositiveInt(arch.linearValueHeadDim ?? config.linear_value_head_dim),
    linearConvKernelDim: parsePositiveInt(arch.linearConvKernelDim ?? config.linear_conv_kernel_dim),
    linearNormMode: parseLinearNormMode(arch.linearNormMode ?? config.linear_norm_mode, config.linear_norm_shared, modelId),
    // Config merge resolves an absent field to the schema default.
    linearNormGateActivation: /** @type {import('../../../config/schema/inference.schema.js').LinearNormGateActivation} */ (
      inf.normalization.linearNormGateActivation),
    // Mechanism programs the manifest's declared closure substitutes for the defaults.
    pairedMechanismReplacements: resolveDeclaredMechanismReplacements(manifest.inference?.execution ?? null),
  };
}
