import { log } from '../../../../debug/index.js';
import { mergeConfig, dumpConfigSources } from '../../../../config/merge.js';
import { validateModelOverrides } from '../../../../config/param-validator.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';
import { appendHeterogeneousAttentionValidation, resolveHeterogeneousAttentionContract } from '../attention/heterogeneous-contract.js';
import { resolvePostNormContract } from '../normalization-contract.js';
import { resolveEmbeddingNormalization } from '../embedding-contract.js';
import {
  PER_LAYER_INPUT_MATERIALIZATION_MODES,
  PER_LAYER_INPUT_ROW_CACHE_MODES,
  PER_LAYER_INPUT_PREFETCH_MODES,
  PER_LAYER_INPUT_GPU_UPLOAD_MODES,
  PER_LAYER_INPUT_HOT_CACHE_MODES,
  PREFILL_CHUNK_SUBMIT_MODES,
} from '../../../../config/schema/execution-v1.schema.js';
import {
  assertSupportedManifestInference,
  validateLayerIntermediateSizesAgainstManifest,
  validateRequiredInferenceFields,
} from './validation.js';

export function resolvePerLayerInputsSession(inferenceConfig, modelId) {
  const sessionConfig = inferenceConfig?.session?.perLayerInputs;
  if (sessionConfig === undefined) {
    throw new Error(
      `Manifest "${modelId}" is missing inference.session.perLayerInputs. ` +
      'Re-convert the model so per-layer input materialization policy is explicit.'
    );
  }
  if (!sessionConfig || typeof sessionConfig !== 'object' || Array.isArray(sessionConfig)) {
    throw new Error(`Manifest "${modelId}" has invalid inference.session.perLayerInputs.`);
  }

  const materialization = sessionConfig.materialization;
  if (!PER_LAYER_INPUT_MATERIALIZATION_MODES.includes(materialization)) {
    throw new Error(
      `Manifest "${modelId}" has invalid inference.session.perLayerInputs.materialization ` +
      `"${String(materialization)}".`
    );
  }

  const rowCache = sessionConfig.rowCache;
  if (!rowCache || typeof rowCache !== 'object' || Array.isArray(rowCache)) {
    throw new Error(`Manifest "${modelId}" is missing inference.session.perLayerInputs.rowCache.`);
  }
  if (!PER_LAYER_INPUT_ROW_CACHE_MODES.includes(rowCache.mode)) {
    throw new Error(
      `Manifest "${modelId}" has invalid inference.session.perLayerInputs.rowCache.mode ` +
      `"${String(rowCache.mode)}".`
    );
  }

  const maxRows = Math.trunc(Number(rowCache.maxRows));
  if (!Number.isFinite(maxRows) || maxRows <= 0) {
    throw new Error(
      `Manifest "${modelId}" requires inference.session.perLayerInputs.rowCache.maxRows ` +
      `to be a positive integer; got ${String(rowCache.maxRows)}.`
    );
  }

  const maxBytes = Math.trunc(Number(rowCache.maxBytes));
  if (!Number.isFinite(maxBytes) || maxBytes <= 0) {
    throw new Error(
      `Manifest "${modelId}" requires inference.session.perLayerInputs.rowCache.maxBytes ` +
      `to be a positive integer; got ${String(rowCache.maxBytes)}.`
    );
  }

  const decodedDtype = String(rowCache.decodedDtype ?? '').toLowerCase();
  if (decodedDtype !== 'f16' && decodedDtype !== 'f32') {
    throw new Error(
      `Manifest "${modelId}" has invalid inference.session.perLayerInputs.rowCache.decodedDtype ` +
      `"${String(rowCache.decodedDtype)}".`
    );
  }

  const prefetch = sessionConfig.prefetch;
  if (!prefetch || typeof prefetch !== 'object' || Array.isArray(prefetch)) {
    throw new Error(`Manifest "${modelId}" is missing inference.session.perLayerInputs.prefetch.`);
  }
  if (!PER_LAYER_INPUT_PREFETCH_MODES.includes(prefetch.mode)) {
    throw new Error(
      `Manifest "${modelId}" has invalid inference.session.perLayerInputs.prefetch.mode ` +
      `"${String(prefetch.mode)}".`
    );
  }

  const rowsAhead = Math.trunc(Number(prefetch.rowsAhead));
  if (!Number.isFinite(rowsAhead) || rowsAhead <= 0) {
    throw new Error(
      `Manifest "${modelId}" requires inference.session.perLayerInputs.prefetch.rowsAhead ` +
      `to be a positive integer; got ${String(prefetch.rowsAhead)}.`
    );
  }

  const gpuUpload = sessionConfig.gpuUpload;
  if (!gpuUpload || typeof gpuUpload !== 'object' || Array.isArray(gpuUpload)) {
    throw new Error(`Manifest "${modelId}" is missing inference.session.perLayerInputs.gpuUpload.`);
  }
  if (!PER_LAYER_INPUT_GPU_UPLOAD_MODES.includes(gpuUpload.mode)) {
    throw new Error(
      `Manifest "${modelId}" has invalid inference.session.perLayerInputs.gpuUpload.mode ` +
      `"${String(gpuUpload.mode)}".`
    );
  }

  const stagingRows = Math.trunc(Number(gpuUpload.stagingRows));
  if (!Number.isFinite(stagingRows) || stagingRows <= 0) {
    throw new Error(
      `Manifest "${modelId}" requires inference.session.perLayerInputs.gpuUpload.stagingRows ` +
      `to be a positive integer; got ${String(gpuUpload.stagingRows)}.`
    );
  }

  const hotCache = sessionConfig.hotCache;
  if (!hotCache || typeof hotCache !== 'object' || Array.isArray(hotCache)) {
    throw new Error(`Manifest "${modelId}" is missing inference.session.perLayerInputs.hotCache.`);
  }
  if (!PER_LAYER_INPUT_HOT_CACHE_MODES.includes(hotCache.mode)) {
    throw new Error(
      `Manifest "${modelId}" has invalid inference.session.perLayerInputs.hotCache.mode ` +
      `"${String(hotCache.mode)}".`
    );
  }

  const hotMaxTokens = Math.trunc(Number(hotCache.maxTokens));
  if (!Number.isFinite(hotMaxTokens) || hotMaxTokens <= 0) {
    throw new Error(
      `Manifest "${modelId}" requires inference.session.perLayerInputs.hotCache.maxTokens ` +
      `to be a positive integer; got ${String(hotCache.maxTokens)}.`
    );
  }

  const hotMaxBytes = Math.trunc(Number(hotCache.maxBytes));
  if (!Number.isFinite(hotMaxBytes) || hotMaxBytes <= 0) {
    throw new Error(
      `Manifest "${modelId}" requires inference.session.perLayerInputs.hotCache.maxBytes ` +
      `to be a positive integer; got ${String(hotCache.maxBytes)}.`
    );
  }

  const hotOutputDtype = String(hotCache.outputDtype ?? '').toLowerCase();
  if (hotOutputDtype !== 'f16' && hotOutputDtype !== 'f32') {
    throw new Error(
      `Manifest "${modelId}" has invalid inference.session.perLayerInputs.hotCache.outputDtype ` +
      `"${String(hotCache.outputDtype)}".`
    );
  }

  return {
    materialization,
    rowCache: {
      mode: rowCache.mode,
      maxRows,
      maxBytes,
      decodedDtype,
    },
    prefetch: {
      mode: prefetch.mode,
      rowsAhead,
    },
    gpuUpload: {
      mode: gpuUpload.mode,
      stagingRows,
    },
    hotCache: {
      mode: hotCache.mode,
      maxTokens: hotMaxTokens,
      maxBytes: hotMaxBytes,
      outputDtype: hotOutputDtype,
    },
  };
}

export function resolveSessionSettings(inferenceConfig, modelId) {
  // All four fields are optional on the manifest today for backwards compatibility;
  // when absent we fall through to the runtime config (getRuntimeConfig()). When
  // present on the manifest, manifest wins unless an explicit runtime profile overrides.
  // The merge layer in src/config/merge.js already applies that precedence — this
  // resolver just validates the manifest-supplied values and normalizes them.
  const session = inferenceConfig?.session;
  const submit = session?.prefillChunkSubmitMode;
  if (submit !== undefined && submit !== null && !PREFILL_CHUNK_SUBMIT_MODES.includes(submit)) {
    throw new Error(
      `Manifest "${modelId}" has invalid inference.session.prefillChunkSubmitMode ` +
      `"${String(submit)}"; expected one of ${PREFILL_CHUNK_SUBMIT_MODES.join(', ')}.`
    );
  }
  const tokenChunk = session?.prefillTokenChunkSize;
  if (
    tokenChunk !== undefined
    && tokenChunk !== null
    && (!Number.isInteger(tokenChunk) || tokenChunk <= 0)
  ) {
    throw new Error(
      `Manifest "${modelId}" has invalid inference.session.prefillTokenChunkSize ` +
      `"${String(tokenChunk)}"; expected null or a positive integer.`
    );
  }
  const skipEmbeddingKVCacheWrites = session?.skipEmbeddingKVCacheWrites;
  if (
    skipEmbeddingKVCacheWrites !== undefined
    && skipEmbeddingKVCacheWrites !== null
    && typeof skipEmbeddingKVCacheWrites !== 'boolean'
  ) {
    throw new Error(`Manifest "${modelId}" has invalid inference.session.skipEmbeddingKVCacheWrites "${String(skipEmbeddingKVCacheWrites)}"; expected boolean.`);
  }
  const flash = session?.useFlashPrefillAttention;
  if (flash !== undefined && flash !== null && typeof flash !== 'boolean') {
    throw new Error(`Manifest "${modelId}" has invalid inference.session.useFlashPrefillAttention "${String(flash)}"; expected boolean.`);
  }
  const largeBatchF16F32FusedGateUp = session?.useLargeBatchF16F32FusedGateUp;
  if (largeBatchF16F32FusedGateUp !== undefined && largeBatchF16F32FusedGateUp !== null && typeof largeBatchF16F32FusedGateUp !== 'boolean') {
    throw new Error(`Manifest "${modelId}" has invalid inference.session.useLargeBatchF16F32FusedGateUp "${String(largeBatchF16F32FusedGateUp)}"; expected boolean.`);
  }
  const wide = session?.useWideTileQ4KPrefill;
  if (wide !== undefined && wide !== null && typeof wide !== 'boolean') {
    throw new Error(`Manifest "${modelId}" has invalid inference.session.useWideTileQ4KPrefill "${String(wide)}"; expected boolean.`);
  }
  const wideDecode = session?.useWideTileQ4KDecode;
  if (wideDecode !== undefined && wideDecode !== null && typeof wideDecode !== 'boolean') {
    throw new Error(`Manifest "${modelId}" has invalid inference.session.useWideTileQ4KDecode "${String(wideDecode)}"; expected boolean.`);
  }
  const sandwichRmsNormPair = session?.useSandwichRMSNormPairFusion;
  if (sandwichRmsNormPair !== undefined && sandwichRmsNormPair !== null && typeof sandwichRmsNormPair !== 'boolean') {
    throw new Error(`Manifest "${modelId}" has invalid inference.session.useSandwichRMSNormPairFusion "${String(sandwichRmsNormPair)}"; expected boolean.`);
  }
  const postFfnNextInputRmsNormPair = session?.usePostFfnNextInputRMSNormPairFusion;
  if (postFfnNextInputRmsNormPair !== undefined && postFfnNextInputRmsNormPair !== null && typeof postFfnNextInputRmsNormPair !== 'boolean') {
    throw new Error(`Manifest "${modelId}" has invalid inference.session.usePostFfnNextInputRMSNormPairFusion "${String(postFfnNextInputRmsNormPair)}"; expected boolean.`);
  }
  const postAttnNormFusedGateUp = session?.usePostAttnNormFusedGateUp;
  if (postAttnNormFusedGateUp !== undefined && postAttnNormFusedGateUp !== null && typeof postAttnNormFusedGateUp !== 'boolean') {
    throw new Error(`Manifest "${modelId}" has invalid inference.session.usePostAttnNormFusedGateUp "${String(postAttnNormFusedGateUp)}"; expected boolean.`);
  }
  const linearAttentionABProjectionFusion = session?.useLinearAttentionABProjectionFusion;
  if (linearAttentionABProjectionFusion !== undefined && linearAttentionABProjectionFusion !== null && typeof linearAttentionABProjectionFusion !== 'boolean') {
    throw new Error(`Manifest "${modelId}" has invalid inference.session.useLinearAttentionABProjectionFusion "${String(linearAttentionABProjectionFusion)}"; expected boolean.`);
  }
  const linearAttentionQKVZProjectionFusion = session?.useLinearAttentionQKVZProjectionFusion;
  if (linearAttentionQKVZProjectionFusion !== undefined && linearAttentionQKVZProjectionFusion !== null && typeof linearAttentionQKVZProjectionFusion !== 'boolean') {
    throw new Error(`Manifest "${modelId}" has invalid inference.session.useLinearAttentionQKVZProjectionFusion "${String(linearAttentionQKVZProjectionFusion)}"; expected boolean.`);
  }
  const linearAttentionFusedDecodeCore = session?.useLinearAttentionFusedDecodeCore;
  if (linearAttentionFusedDecodeCore !== undefined && linearAttentionFusedDecodeCore !== null && typeof linearAttentionFusedDecodeCore !== 'boolean') {
    throw new Error(`Manifest "${modelId}" has invalid inference.session.useLinearAttentionFusedDecodeCore "${String(linearAttentionFusedDecodeCore)}"; expected boolean.`);
  }
  const wideTileResidualFusion = session?.useWideTileResidualFusion;
  if (wideTileResidualFusion !== undefined && wideTileResidualFusion !== null && typeof wideTileResidualFusion !== 'boolean') {
    throw new Error(`Manifest "${modelId}" has invalid inference.session.useWideTileResidualFusion "${String(wideTileResidualFusion)}"; expected boolean.`);
  }
  const fusedRmsnormWideTile = session?.useFusedRmsnormWideTile;
  if (fusedRmsnormWideTile !== undefined && fusedRmsnormWideTile !== null && typeof fusedRmsnormWideTile !== 'boolean') {
    throw new Error(`Manifest "${modelId}" has invalid inference.session.useFusedRmsnormWideTile "${String(fusedRmsnormWideTile)}"; expected boolean.`);
  }
  const fusedQKVSplitQKNorm = session?.useFusedQKVSplitQKNorm;
  if (fusedQKVSplitQKNorm !== undefined && fusedQKVSplitQKNorm !== null && typeof fusedQKVSplitQKNorm !== 'boolean') {
    throw new Error(`Manifest "${modelId}" has invalid inference.session.useFusedQKVSplitQKNorm "${String(fusedQKVSplitQKNorm)}"; expected boolean.`);
  }
  const fusedQKVSplitQKNormRoPE = session?.useFusedQKVSplitQKNormRoPE;
  if (fusedQKVSplitQKNormRoPE !== undefined && fusedQKVSplitQKNormRoPE !== null && typeof fusedQKVSplitQKNormRoPE !== 'boolean') {
    throw new Error(`Manifest "${modelId}" has invalid inference.session.useFusedQKVSplitQKNormRoPE "${String(fusedQKVSplitQKNormRoPE)}"; expected boolean.`);
  }
  const retain = session?.retainQ4KMaterialization;
  if (retain !== undefined && retain !== null && typeof retain !== 'boolean') {
    throw new Error(`Manifest "${modelId}" has invalid inference.session.retainQ4KMaterialization "${String(retain)}"; expected boolean.`);
  }
  const lmHeadArgmaxQ4K = session?.lmHeadArgmaxQ4K;
  if (lmHeadArgmaxQ4K !== undefined && lmHeadArgmaxQ4K !== null) {
    if (typeof lmHeadArgmaxQ4K !== 'object' || Array.isArray(lmHeadArgmaxQ4K)) {
      throw new Error(`Manifest "${modelId}" has invalid inference.session.lmHeadArgmaxQ4K; expected object or null.`);
    }
    const fullBlockFastPath = lmHeadArgmaxQ4K.useFullBlockFastPath;
    if (fullBlockFastPath !== undefined && typeof fullBlockFastPath !== 'boolean') {
      throw new Error(
        `Manifest "${modelId}" has invalid inference.session.lmHeadArgmaxQ4K.useFullBlockFastPath ` +
        `"${String(fullBlockFastPath)}"; expected boolean.`
      );
    }
    const colsPerWorkgroup = lmHeadArgmaxQ4K.colsPerWorkgroup;
    if (colsPerWorkgroup !== undefined && (!Number.isInteger(colsPerWorkgroup) || colsPerWorkgroup <= 0)) {
      throw new Error(
        `Manifest "${modelId}" has invalid inference.session.lmHeadArgmaxQ4K.colsPerWorkgroup ` +
        `"${String(colsPerWorkgroup)}"; expected positive integer.`
      );
    }
    const threadsPerCol = lmHeadArgmaxQ4K.threadsPerCol;
    if (threadsPerCol !== undefined && (!Number.isInteger(threadsPerCol) || threadsPerCol <= 0)) {
      throw new Error(
        `Manifest "${modelId}" has invalid inference.session.lmHeadArgmaxQ4K.threadsPerCol ` +
        `"${String(threadsPerCol)}"; expected positive integer.`
      );
    }
  }
  const attentionDecodeOnline = session?.attentionDecodeOnline;
  if (attentionDecodeOnline !== undefined && attentionDecodeOnline !== null) {
    if (typeof attentionDecodeOnline !== 'object' || Array.isArray(attentionDecodeOnline)) {
      throw new Error(`Manifest "${modelId}" has invalid inference.session.attentionDecodeOnline; expected object or null.`);
    }
    const workgroupSize = attentionDecodeOnline.workgroupSize;
    if (workgroupSize !== undefined && workgroupSize !== 128 && workgroupSize !== 256) {
      throw new Error(
        `Manifest "${modelId}" has invalid inference.session.attentionDecodeOnline.workgroupSize ` +
        `"${String(workgroupSize)}"; expected 128 or 256.`
      );
    }
    const directKv = attentionDecodeOnline.useDirectContiguousKVLayout;
    if (directKv !== undefined && typeof directKv !== 'boolean') {
      throw new Error(
        `Manifest "${modelId}" has invalid inference.session.attentionDecodeOnline.useDirectContiguousKVLayout ` +
        `"${String(directKv)}"; expected boolean.`
      );
    }
    const outputGateFusion = attentionDecodeOnline.useOutputGateFusion;
    if (outputGateFusion !== undefined && typeof outputGateFusion !== 'boolean') {
      throw new Error(
        `Manifest "${modelId}" has invalid inference.session.attentionDecodeOnline.useOutputGateFusion ` +
        `"${String(outputGateFusion)}"; expected boolean.`
      );
    }
  }
  return {
    prefillChunkSubmitMode: submit ?? null,
    prefillTokenChunkSize: tokenChunk ?? null,
    skipEmbeddingKVCacheWrites: skipEmbeddingKVCacheWrites ?? null,
    useFlashPrefillAttention: flash ?? null,
    useLargeBatchF16F32FusedGateUp: largeBatchF16F32FusedGateUp ?? null,
    useWideTileQ4KPrefill: wide ?? null,
    useWideTileQ4KDecode: wideDecode ?? null,
    useSandwichRMSNormPairFusion: sandwichRmsNormPair ?? null,
    usePostFfnNextInputRMSNormPairFusion: postFfnNextInputRmsNormPair ?? null,
    usePostAttnNormFusedGateUp: postAttnNormFusedGateUp ?? null,
    useLinearAttentionABProjectionFusion: linearAttentionABProjectionFusion ?? null,
    useLinearAttentionQKVZProjectionFusion: linearAttentionQKVZProjectionFusion ?? null,
    useLinearAttentionFusedDecodeCore: linearAttentionFusedDecodeCore ?? null,
    useWideTileResidualFusion: wideTileResidualFusion ?? null,
    useFusedRmsnormWideTile: fusedRmsnormWideTile ?? null,
    useFusedQKVSplitQKNorm: fusedQKVSplitQKNorm ?? null,
    useFusedQKVSplitQKNormRoPE: fusedQKVSplitQKNormRoPE ?? null,
    retainQ4KMaterialization: retain ?? null,
    lmHeadArgmaxQ4K: lmHeadArgmaxQ4K ?? null,
    attentionDecodeOnline: attentionDecodeOnline ?? null,
  };
}

export function resolveVisionConfig(rawConfig, manifest) {
  const vc = rawConfig?.vision_config ?? manifest?.config?.vision_config;
  if (!vc || typeof vc !== 'object') {
    log.debug(
      'Config',
      `Vision config not present for model "${manifest?.modelId ?? 'unknown'}"; vision pipeline disabled.`
    );
    return null;
  }
  const modelId = manifest?.modelId ?? 'unknown';
  const resolveRequiredVisionField = (keys, label) => {
    for (const key of keys) {
      if (vc[key] !== undefined) {
        return vc[key];
      }
    }
    throw new Error(
      `Manifest "${modelId}" is missing vision_config.${label}. ` +
      'Re-convert the model with explicit vision config metadata.'
    );
  };
  const resolveRequiredPositiveInteger = (keys, label) => {
    const value = resolveRequiredVisionField(keys, label);
    const number = Number(value);
    if (!Number.isFinite(number) || number <= 0 || Math.floor(number) !== number) {
      throw new Error(
        `Manifest "${modelId}" has invalid vision_config.${label}=${JSON.stringify(value)}. ` +
        'Expected a positive integer.'
      );
    }
    return Math.trunc(number);
  };
  const resolveRequiredNonNegativeInteger = (keys, label) => {
    const value = resolveRequiredVisionField(keys, label);
    const number = Number(value);
    if (!Number.isFinite(number) || number < 0 || Math.floor(number) !== number) {
      throw new Error(
        `Manifest "${modelId}" has invalid vision_config.${label}=${JSON.stringify(value)}. ` +
        'Expected a non-negative integer.'
      );
    }
    return Math.trunc(number);
  };
  const resolveRequiredPositiveNumber = (value, label) => {
    const number = Number(value);
    if (!Number.isFinite(number) || number <= 0) {
      throw new Error(
        `Manifest "${modelId}" has invalid ${label}=${JSON.stringify(value)}. ` +
        'Expected a positive number.'
      );
    }
    return number;
  };
  const visionArchitecture = String(resolveRequiredVisionField(['vision_architecture'], 'vision_architecture')).trim();
  if (visionArchitecture !== 'gemma4' && visionArchitecture !== 'qwen3vl' && visionArchitecture !== 'glmocr') {
    throw new Error(
      `Manifest "${modelId}" has unsupported vision_config.vision_architecture="${visionArchitecture}". ` +
      'Supported: "gemma4", "qwen3vl", "glmocr".'
    );
  }

  if (visionArchitecture === 'gemma4') {
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
        : resolveRequiredPositiveNumber(ropeParameters.rope_theta, 'vision_config.rope_parameters.rope_theta'),
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

  if (visionArchitecture === 'glmocr') {
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

export function resolveAudioConfig(rawConfig, manifest) {
  const ac = rawConfig?.audio_config ?? manifest?.config?.audio_config;
  if (!ac || typeof ac !== 'object') {
    log.debug(
      'Config',
      `Audio config not present for model "${manifest?.modelId ?? 'unknown'}"; audio pipeline disabled.`
    );
    return null;
  }
  const modelId = manifest?.modelId ?? 'unknown';
  const resolveRequiredPositiveInteger = (value, label) => {
    const number = Number(value);
    if (!Number.isFinite(number) || number <= 0 || Math.floor(number) !== number) {
      throw new Error(
        `Manifest "${modelId}" has invalid audio_config.${label}=${JSON.stringify(value)}. ` +
        'Expected a positive integer.'
      );
    }
    return Math.trunc(number);
  };
  const resolveRequiredNonNegativeInteger = (value, label) => {
    const number = Number(value);
    if (!Number.isFinite(number) || number < 0 || Math.floor(number) !== number) {
      throw new Error(
        `Manifest "${modelId}" has invalid audio_config.${label}=${JSON.stringify(value)}. ` +
        'Expected a non-negative integer.'
      );
    }
    return Math.trunc(number);
  };
  const resolveRequiredPositiveNumber = (value, label) => {
    const number = Number(value);
    if (!Number.isFinite(number) || number <= 0) {
      throw new Error(
        `Manifest "${modelId}" has invalid audio_config.${label}=${JSON.stringify(value)}. ` +
        'Expected a positive number.'
      );
    }
    return number;
  };
  const resolveRequiredFiniteNumber = (value, label) => {
    const number = Number(value);
    if (!Number.isFinite(number)) {
      throw new Error(
        `Manifest "${modelId}" has invalid audio_config.${label}=${JSON.stringify(value)}. ` +
        'Expected a finite number.'
      );
    }
    return number;
  };
  const resolveRequiredString = (value, label) => {
    if (typeof value !== 'string' || value.trim().length === 0) {
      throw new Error(
        `Manifest "${modelId}" has invalid audio_config.${label}=${JSON.stringify(value)}. ` +
        'Expected a non-empty string.'
      );
    }
    return value.trim();
  };
  const audioArchitecture = String(ac.audio_architecture ?? '').trim();
  if (audioArchitecture !== 'gemma4') {
    throw new Error(
      `Manifest "${modelId}" has unsupported audio_config.audio_architecture="${audioArchitecture}". ` +
      'Supported: "gemma4".'
    );
  }

  const depth = resolveRequiredNonNegativeInteger(ac.num_hidden_layers, 'num_hidden_layers');
  const isEncoderFree = (depth === 0);

  const hiddenSize = resolveRequiredPositiveInteger(ac.hidden_size, 'hidden_size');
  const numAttentionHeads = resolveRequiredPositiveInteger(ac.num_attention_heads, 'num_attention_heads');
  const headDim = Math.trunc(hiddenSize / numAttentionHeads);

  if (!isEncoderFree) {
    if (!Array.isArray(ac.subsampling_conv_channels) || ac.subsampling_conv_channels.length < 1) {
      throw new Error(
        `Manifest "${modelId}" is missing audio_config.subsampling_conv_channels array.`
      );
    }
  }

  return {
    audioArchitecture,
    depth,
    hiddenSize,
    numAttentionHeads,
    headDim,
    convKernelSize: resolveRequiredPositiveInteger(ac.conv_kernel_size, 'conv_kernel_size'),
    subsamplingConvChannels: ac.subsampling_conv_channels.map(Number),
    outputProjDims: resolveRequiredPositiveInteger(ac.output_proj_dims, 'output_proj_dims'),
    attentionContextLeft: resolveRequiredPositiveInteger(ac.attention_context_left, 'attention_context_left'),
    attentionContextRight: resolveRequiredNonNegativeInteger(ac.attention_context_right, 'attention_context_right'),
    attentionChunkSize: resolveRequiredPositiveInteger(ac.attention_chunk_size, 'attention_chunk_size'),
    attentionLogitCap: resolveRequiredPositiveNumber(ac.attention_logit_cap, 'attention_logit_cap'),
    attentionInvalidLogitsValue: resolveRequiredFiniteNumber(ac.attention_invalid_logits_value, 'attention_invalid_logits_value'),
    residualWeight: resolveRequiredPositiveNumber(ac.residual_weight, 'residual_weight'),
    rmsNormEps: resolveRequiredPositiveNumber(ac.rms_norm_eps, 'rms_norm_eps'),
    hiddenAct: resolveRequiredString(ac.hidden_act, 'hidden_act'),
    useClippedLinears: ac.use_clipped_linears === true,
    audioTokenId: rawConfig?.audio_token_id ?? manifest?.audio_token_id ?? null,
  };
}
