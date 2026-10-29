// Per-architecture normalization of manifest vision_config. The caller owns
// field resolution helpers and the unsupported-architecture error.
/** @param {import('./vision-normalizers.js').VisionConfigNormalizerContext} context */
function normalizeGemma4VisionConfig({ vc, modelId, rawConfig, manifest, resolveRequiredVisionField, resolveRequiredPositiveInteger, resolveRequiredNonNegativeInteger, resolveRequiredPositiveNumber, visionArchitecture }) {
  const depth = resolveRequiredNonNegativeInteger(['depth', 'num_hidden_layers'], 'num_hidden_layers');
  const isEncoderFree = (depth === 0);

  const hiddenSize = resolveRequiredPositiveInteger(['hidden_size'], 'hidden_size');
  const numHeads = isEncoderFree
    ? (vc.num_heads !== undefined || vc.num_attention_heads !== undefined ? resolveRequiredPositiveInteger(['num_heads', 'num_attention_heads'], 'num_attention_heads') : 1)
    : resolveRequiredPositiveInteger(['num_heads', 'num_attention_heads'], 'num_attention_heads');
  const ropeParameters = vc.rope_parameters;
  if (!ropeParameters || typeof ropeParameters !== 'object') {
    if (!isEncoderFree) {
      throw new Error(
        `Manifest "${modelId}" is missing vision_config.rope_parameters. ` +
        'Re-convert the model with explicit Gemma 4 vision RoPE metadata.'
      );
    }
  }
  const hiddenActivation = String(resolveRequiredVisionField(['hidden_activation'], 'hidden_activation')).trim();
  if (hiddenActivation !== 'gelu' && hiddenActivation !== 'gelu_pytorch_tanh') {
    throw new Error(
      `Manifest "${modelId}" has unsupported Gemma 4 vision hidden_activation="${hiddenActivation}". ` +
      'Supported values: "gelu", "gelu_pytorch_tanh".'
    );
  }
  if (vc.standardize === true) {
    throw new Error(
      `Manifest "${modelId}" enables vision_config.standardize, but Gemma 4 runtime preprocessing does not support it yet.`
    );
  }
  if (vc.use_clipped_linears !== true) {
    throw new Error(
      `Manifest "${modelId}" requires vision_config.use_clipped_linears=true for Gemma 4 vision weights.`
    );
  }

  return {
    depth,
    hiddenSize,
    intermediateSize: isEncoderFree
      ? (vc.intermediate_size !== undefined ? resolveRequiredPositiveInteger(['intermediate_size'], 'intermediate_size') : 1)
      : resolveRequiredPositiveInteger(['intermediate_size'], 'intermediate_size'),
    numHeads,
    numKeyValueHeads: isEncoderFree
      ? (vc.num_key_value_heads !== undefined ? resolveRequiredPositiveInteger(['num_key_value_heads'], 'num_key_value_heads') : 1)
      : resolveRequiredPositiveInteger(['num_key_value_heads'], 'num_key_value_heads'),
    headDim: isEncoderFree
      ? (vc.head_dim !== undefined || vc.global_head_dim !== undefined ? resolveRequiredPositiveInteger(['head_dim', 'global_head_dim'], 'head_dim') : 1)
      : resolveRequiredPositiveInteger(['head_dim', 'global_head_dim'], 'head_dim'),
    outHiddenSize: vc.out_hidden_size ?? vc.output_proj_dims ?? null,
    patchSize: resolveRequiredPositiveInteger(['patch_size'], 'patch_size'),
    poolingKernelSize: resolveRequiredPositiveInteger(['pooling_kernel_size'], 'pooling_kernel_size'),
    spatialMergeSize: vc.spatial_merge_size ?? null,
    temporalPatchSize: vc.temporal_patch_size ?? null,
    positionEmbeddingSize: resolveRequiredPositiveInteger(['position_embedding_size'], 'position_embedding_size'),
    defaultOutputLength: resolveRequiredPositiveInteger(['default_output_length'], 'default_output_length'),
    ropeTheta: isEncoderFree && (!ropeParameters || ropeParameters.rope_theta === undefined)
      ? 10000
      : resolveRequiredPositiveNumber(/** @type {NonNullable<typeof ropeParameters>} */ (ropeParameters).rope_theta, 'vision_config.rope_parameters.rope_theta'),
    eps: resolveRequiredPositiveNumber(
      resolveRequiredVisionField(['eps', 'rms_norm_eps'], 'rms_norm_eps'),
      'vision_config.rms_norm_eps'
    ),
    hiddenActivation,
    standardize: false,
    useClippedLinears: true,
    deepstackVisualIndexes: [],
    imageTokenId: rawConfig?.image_token_id ?? manifest?.image_token_id ?? null,
    visionArchitecture,
    softTokenBudgetTiers: Array.isArray(vc.soft_token_budget_tiers)
      ? vc.soft_token_budget_tiers.map(Number).filter((n) => Number.isFinite(n) && n > 0)
      : [70, 140, 280, 560, 1120],
  };
}

/** @param {import('./vision-normalizers.js').VisionConfigNormalizerContext} context */
function normalizeGlmOcrVisionConfig({ vc, modelId, rawConfig, manifest, resolveRequiredVisionField, resolveRequiredPositiveInteger, resolveRequiredPositiveNumber, visionArchitecture }) {
  const hiddenSize = resolveRequiredPositiveInteger(['hidden_size'], 'hidden_size');
  const numHeads = resolveRequiredPositiveInteger(['num_heads', 'num_attention_heads'], 'num_heads');
  if (hiddenSize % numHeads !== 0) {
    throw new Error(
      `Manifest "${modelId}" has incompatible GLM-OCR vision geometry: ` +
      `hidden_size=${hiddenSize} is not divisible by num_heads=${numHeads}.`
    );
  }
  const hiddenActivation = String(
    resolveRequiredVisionField(['hidden_activation', 'hidden_act'], 'hidden_activation')
  ).trim();
  if (hiddenActivation !== 'silu') {
    throw new Error(
      `Manifest "${modelId}" has unsupported GLM-OCR vision hidden_activation="${hiddenActivation}". ` +
      'The current architecture contract requires "silu".'
    );
  }
  const spatialMergeSize = resolveRequiredPositiveInteger(
    ['spatial_merge_size', 'merge_size'],
    'spatial_merge_size'
  );
  const normalization = vc.normalization;
  if (!normalization || typeof normalization !== 'object') {
    throw new Error(
      `Manifest "${modelId}" is missing vision_config.normalization. ` +
      'Re-convert the GLM-OCR model with its pinned image-processor values.'
    );
  }
  if (!Array.isArray(normalization.mean) || normalization.mean.length !== 3) {
    throw new Error(
      `Manifest "${modelId}" requires vision_config.normalization.mean to contain exactly 3 values.`
    );
  }
  if (!Array.isArray(normalization.std) || normalization.std.length !== 3) {
    throw new Error(
      `Manifest "${modelId}" requires vision_config.normalization.std to contain exactly 3 values.`
    );
  }
  return {
    depth: resolveRequiredPositiveInteger(['depth', 'num_hidden_layers'], 'depth'),
    hiddenSize,
    intermediateSize: resolveRequiredPositiveInteger(['intermediate_size'], 'intermediate_size'),
    numHeads,
    numKeyValueHeads: numHeads,
    headDim: hiddenSize / numHeads,
    outHiddenSize: resolveRequiredPositiveInteger(['out_hidden_size'], 'out_hidden_size'),
    patchSize: resolveRequiredPositiveInteger(['patch_size'], 'patch_size'),
    poolingKernelSize: spatialMergeSize,
    spatialMergeSize,
    temporalPatchSize: resolveRequiredPositiveInteger(['temporal_patch_size'], 'temporal_patch_size'),
    positionEmbeddingSize: null,
    defaultOutputLength: resolveRequiredPositiveInteger(['default_output_length'], 'default_output_length'),
    ropeTheta: resolveRequiredPositiveNumber(
      resolveRequiredVisionField(['rope_theta'], 'rope_theta'),
      'vision_config.rope_theta'
    ),
    eps: resolveRequiredPositiveNumber(
      resolveRequiredVisionField(['eps', 'rms_norm_eps'], 'rms_norm_eps'),
      'vision_config.rms_norm_eps'
    ),
    hiddenActivation,
    standardize: false,
    useClippedLinears: false,
    minPixels: resolveRequiredPositiveInteger(['min_pixels'], 'min_pixels'),
    maxPixels: resolveRequiredPositiveInteger(['max_pixels'], 'max_pixels'),
    normalization,
    inChannels: resolveRequiredPositiveInteger(['in_channels'], 'in_channels'),
    mergerIntermediateSize: resolveRequiredPositiveInteger(
      ['merger_intermediate_size'],
      'merger_intermediate_size'
    ),
    downsampleKernelSize: resolveRequiredPositiveInteger(
      ['downsample_kernel_size'],
      'downsample_kernel_size'
    ),
    deepstackVisualIndexes: [],
    imageTokenId: rawConfig?.image_token_id ?? manifest?.image_token_id ?? null,
    visionArchitecture,
    softTokenBudgetTiers: Array.isArray(vc.soft_token_budget_tiers)
      ? vc.soft_token_budget_tiers.map(Number).filter((n) => Number.isFinite(n) && n > 0)
      : [384, 768, 1536, 3072, 6144],
  };
}

/** @param {import('./vision-normalizers.js').VisionConfigNormalizerContext} context */
function normalizeQwen3VlVisionConfig({ vc, modelId, rawConfig, manifest, resolveRequiredVisionField, resolveRequiredPositiveInteger, resolveRequiredPositiveNumber, visionArchitecture }) {
  const hiddenSize = resolveRequiredPositiveInteger(['hidden_size'], 'hidden_size');
  const intermediateSize = resolveRequiredPositiveInteger(['intermediate_size'], 'intermediate_size');
  const numHeads = resolveRequiredPositiveInteger(['num_heads', 'num_attention_heads'], 'num_heads');
  const outHiddenSize = resolveRequiredPositiveInteger(['out_hidden_size', 'output_proj_dims'], 'out_hidden_size');
  const patchSize = resolveRequiredPositiveInteger(['patch_size'], 'patch_size');
  const spatialMergeSize = resolveRequiredPositiveInteger(['spatial_merge_size', 'merge_size'], 'spatial_merge_size');
  const temporalPatchSize = resolveRequiredPositiveInteger(['temporal_patch_size'], 'temporal_patch_size');
  const eps = resolveRequiredPositiveNumber(
    resolveRequiredVisionField(['eps', 'rms_norm_eps'], 'eps'),
    'vision_config.eps'
  );
  const hiddenActivation = String(
    resolveRequiredVisionField(['hidden_activation', 'hidden_act'], 'hidden_activation')
  ).trim();
  const minPixels = resolveRequiredPositiveInteger(['min_pixels'], 'min_pixels');
  const maxPixels = resolveRequiredPositiveInteger(['max_pixels'], 'max_pixels');
  const normalization = vc.normalization;
  if (!normalization || typeof normalization !== 'object') {
    throw new Error(
      `Manifest "${modelId}" is missing vision_config.normalization. ` +
      'Re-convert the model with explicit normalization metadata.'
    );
  }
  if (!Array.isArray(normalization.mean) || normalization.mean.length !== 3) {
    throw new Error(
      `Manifest "${modelId}" requires vision_config.normalization.mean to contain exactly 3 values.`
    );
  }
  if (!Array.isArray(normalization.std) || normalization.std.length !== 3) {
    throw new Error(
      `Manifest "${modelId}" requires vision_config.normalization.std to contain exactly 3 values.`
    );
  }
  return {
    depth: resolveRequiredPositiveInteger(['depth', 'num_hidden_layers'], 'depth'),
    hiddenSize,
    intermediateSize,
    numHeads,
    numKeyValueHeads: resolveRequiredPositiveInteger(['num_key_value_heads'], 'num_key_value_heads'),
    headDim: resolveRequiredPositiveInteger(['head_dim'], 'head_dim'),
    outHiddenSize,
    patchSize,
    poolingKernelSize: resolveRequiredPositiveInteger(['pooling_kernel_size'], 'pooling_kernel_size'),
    spatialMergeSize,
    temporalPatchSize,
    positionEmbeddingSize: vc.position_embedding_size ?? null,
    defaultOutputLength: vc.default_output_length ?? null,
    ropeTheta: vc.rope_parameters?.rope_theta ?? null,
    eps,
    hiddenActivation,
    standardize: vc.standardize === true,
    useClippedLinears: vc.use_clipped_linears === true,
    minPixels,
    maxPixels,
    normalization,
    deepstackVisualIndexes: Array.isArray(vc.deepstack_visual_indexes) ? vc.deepstack_visual_indexes : [],
    imageTokenId: rawConfig?.image_token_id ?? manifest?.image_token_id ?? null,
    visionArchitecture,
  };
}

// Manifest vision_config.vision_architecture -> config normalizer.
/** @type {Readonly<Record<string, import('./vision-normalizers.js').VisionConfigNormalizer>>} */
const VISION_CONFIG_NORMALIZERS = Object.freeze({
  gemma4: normalizeGemma4VisionConfig,
  qwen3vl: normalizeQwen3VlVisionConfig,
  glmocr: normalizeGlmOcrVisionConfig,
});

/** @type {import('./vision-normalizers.js').resolveVisionConfigNormalizer} */
export function resolveVisionConfigNormalizer(visionArchitecture) {
  return typeof visionArchitecture === 'string' && Object.hasOwn(VISION_CONFIG_NORMALIZERS, visionArchitecture)
    ? VISION_CONFIG_NORMALIZERS[visionArchitecture]
    : null;
}
