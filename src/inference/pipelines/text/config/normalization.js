import { log } from '../../../../debug/index.js';
import { mergeConfig, dumpConfigSources } from '../../../../config/merge.js';
import { validateModelOverrides, validateUnsupportedSessionOptions } from '../../../../config/param-validator.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';
import { appendHeterogeneousAttentionValidation, resolveHeterogeneousAttentionContract } from '../../../../config/heterogeneous-attention-contract.js';
import { resolvePostNormContract } from '../normalization-contract.js';
import { resolveEmbeddingNormalization } from '../embedding-contract.js';
import { resolveVisionConfigNormalizer } from './vision-normalizers.js';
import {
  PER_LAYER_INPUT_MATERIALIZATION_MODES,
  PER_LAYER_INPUT_ROW_CACHE_MODES,
  PER_LAYER_INPUT_PREFETCH_MODES,
  PER_LAYER_INPUT_GPU_UPLOAD_MODES,
  PER_LAYER_INPUT_HOT_CACHE_MODES,
  PREFILL_CHUNK_SUBMIT_MODES,
} from '../../../../config/schema/execution-v1.schema.js';
import { assertSupportedManifestInference, validateLayerIntermediateSizesAgainstManifest, validateRequiredInferenceFields } from '../../../../config/manifest-inference-validation.js';

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
  validateUnsupportedSessionOptions(inferenceConfig?.session, `Manifest ${modelId} inference.session`);
  // All four fields are optional on the manifest today for backwards compatibility;
  // when absent we fall through to the runtime config (getRuntimeConfig()). When
  // present on the manifest, manifest wins unless an explicit runtime profile overrides.
  // The merge layer in src/config/merge.js already applies that precedence — this
  // resolver just validates the manifest-supplied values and normalizes them.
  const session = inferenceConfig?.session;
  for (const field of ['usePreparedDecodeWorkspace', 'useLinearAttentionRegisterPrefill', 'useLinearAttentionRegisterDecode']) {
    if (session?.[field] !== undefined && typeof session[field] !== 'boolean') {
      throw new Error(`Manifest "${modelId}" requires boolean inference.session.${field}.`);
    }
  }
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
  const registerPrefill = session?.useLinearAttentionRegisterPrefill;
  const registerDecode = session?.useLinearAttentionRegisterDecode;
  const linearAttentionFusedDecodeCore = session?.useLinearAttentionFusedDecodeCore;
  if (linearAttentionFusedDecodeCore !== undefined && linearAttentionFusedDecodeCore !== null && typeof linearAttentionFusedDecodeCore !== 'boolean') {
    throw new Error(`Manifest "${modelId}" has invalid inference.session.useLinearAttentionFusedDecodeCore "${String(linearAttentionFusedDecodeCore)}"; expected boolean.`);
  }
  const intermediatePrecision = session?.intermediatePrecision;
  if (intermediatePrecision != null && intermediatePrecision !== 'f32' && intermediatePrecision !== 'two-f32') {
    throw new Error(`Manifest "${modelId}" has invalid inference.session.intermediatePrecision; expected f32 or two-f32.`);
  }
  const linearRecurrentStatePrecision = session?.linearRecurrentStatePrecision;
  if (linearRecurrentStatePrecision != null && linearRecurrentStatePrecision !== 'f32' && linearRecurrentStatePrecision !== 'two-f32') {
    throw new Error(`Manifest "${modelId}" has invalid inference.session.linearRecurrentStatePrecision; expected f32 or two-f32.`);
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
    usePreparedDecodeWorkspace: session?.usePreparedDecodeWorkspace ?? null,
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
    useLinearAttentionRegisterPrefill: registerPrefill ?? null,
    useLinearAttentionRegisterDecode: registerDecode ?? null,
    linearRecurrentStatePrecision: linearRecurrentStatePrecision ?? null,
    intermediatePrecision: intermediatePrecision ?? null,
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
  const normalize = resolveVisionConfigNormalizer(visionArchitecture);
  if (!normalize) {
    throw new Error(
      `Manifest "${modelId}" has unsupported vision_config.vision_architecture="${visionArchitecture}". ` +
      'Supported: "gemma4", "qwen3vl", "glmocr".'
    );
  }
  return normalize({
    vc,
    modelId,
    rawConfig,
    manifest,
    visionArchitecture,
    resolveRequiredVisionField,
    resolveRequiredPositiveInteger,
    resolveRequiredNonNegativeInteger,
    resolveRequiredPositiveNumber,
  });
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
    featureExtractor: resolveAudioFeatureExtractor(ac.feature_extractor, modelId),
  };
}

// The log-mel front end is part of the model contract, not a runtime constant:
// the encoder's first convolution expects exactly the reference mel geometry.
// Manifests carry it as audio_config.feature_extractor using the Hugging Face
// feature-extractor field names; absent fields that mean "feature off" resolve
// to off (preemphasis 0, no overdrive, unit scale, no per-bin normalization).
export function resolveAudioFeatureExtractor(raw, modelId = 'unknown') {
  if (raw == null) return null;
  if (typeof raw !== 'object' || Array.isArray(raw)) {
    throw new Error(`Manifest "${modelId}" has invalid audio_config.feature_extractor; expected an object.`);
  }
  const label = (field) => `audio_config.feature_extractor.${field}`;
  const number = (field, { min = -Infinity, positive = false, integer = false, optional = false, fallback = null } = {}) => {
    const value = raw[field];
    if (value == null) {
      if (optional) return fallback;
      throw new Error(`Manifest "${modelId}" is missing ${label(field)}.`);
    }
    const parsed = Number(value);
    if (!Number.isFinite(parsed) || parsed < min || (positive && parsed <= 0) || (integer && !Number.isInteger(parsed))) {
      throw new Error(`Manifest "${modelId}" has invalid ${label(field)}=${JSON.stringify(value)}.`);
    }
    return parsed;
  };
  const flag = (field, fallback) => {
    const value = raw[field];
    if (value == null) return fallback;
    if (typeof value !== 'boolean') {
      throw new Error(`Manifest "${modelId}" has invalid ${label(field)}=${JSON.stringify(value)}; expected a boolean.`);
    }
    return value;
  };
  const perBin = (field, featureSize) => {
    const value = raw[field];
    if (value == null) return null;
    if (!Array.isArray(value) || value.length !== featureSize || !value.every((entry) => Number.isFinite(Number(entry)))) {
      throw new Error(`Manifest "${modelId}" has invalid ${label(field)}; expected ${featureSize} finite numbers.`);
    }
    return Float32Array.from(value, Number);
  };
  const featureSize = number('feature_size', { positive: true, integer: true });
  const samplingRate = number('sampling_rate', { positive: true, integer: true });
  const frameLengthMs = number('frame_length_ms', { positive: true });
  const hopLengthMs = number('hop_length_ms', { positive: true });
  const frameLength = Math.round((samplingRate * frameLengthMs) / 1000);
  const hopLength = Math.round((samplingRate * hopLengthMs) / 1000);
  if (frameLength < 2 || hopLength < 1) {
    throw new Error(`Manifest "${modelId}" audio_config.feature_extractor frame/hop resolve to ${frameLength}/${hopLength} samples.`);
  }
  const fftOverdrive = flag('fft_overdrive', false);
  const fftLength = (2 ** Math.ceil(Math.log2(frameLength))) * (fftOverdrive ? 2 : 1);
  const minFrequency = number('min_frequency', { min: 0 });
  const maxFrequency = number('max_frequency', { positive: true });
  if (maxFrequency <= minFrequency || maxFrequency > samplingRate / 2) {
    throw new Error(`Manifest "${modelId}" audio_config.feature_extractor frequency range ${minFrequency}-${maxFrequency} Hz is invalid for ${samplingRate} Hz audio.`);
  }
  const dither = number('dither', { min: 0, optional: true, fallback: 0 });
  if (dither > 0) {
    throw new Error(`Manifest "${modelId}" audio_config.feature_extractor.dither=${dither} requests random dither; the runtime front end is deterministic and supports dither 0 only.`);
  }
  return {
    featureSize,
    samplingRate,
    frameLength,
    hopLength,
    fftLength,
    fftOverdrive,
    minFrequency,
    maxFrequency,
    preemphasis: number('preemphasis', { min: 0, optional: true, fallback: 0 }),
    preemphasisHtkFlavor: flag('preemphasis_htk_flavor', true),
    melFloor: number('mel_floor', { positive: true }),
    inputScaleFactor: number('input_scale_factor', { positive: true, optional: true, fallback: 1 }),
    perBinMean: perBin('per_bin_mean', featureSize),
    perBinStddev: perBin('per_bin_stddev', featureSize),
  };
}
