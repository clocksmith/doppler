

import {
  // Constants
  SHARD_SIZE as SCHEMA_SHARD_SIZE,
  RDRR_VERSION as SCHEMA_RDRR_VERSION,
  ConversionStage as SchemaConversionStage,
  DEFAULT_MANIFEST_INFERENCE,
  formatBytes,
} from '../config/schema/index.js';

import {
  classifyTensorRole,
  generateShardFilename,
  resolveTensorGroup,
  resolveTensorRole,
} from '../formats/rdrr/index.js';
import { log } from '../debug/index.js';
import {
  getInferenceLayerPatternContractArtifact,
  selectRuleValue,
} from '../rules/rule-registry.js';
import {
  createConverterConfig,
} from '../config/index.js';
import { buildExecutionContractArtifact } from '../config/execution-contract-check.js';
import { buildManifestRequiredInferenceFieldsArtifact } from '../config/required-inference-fields-contract-check.js';
import { resolveEosTokenId } from './tokenizer-special-tokens.js';
import { inferBundledTokenizerBehaviorFlags } from '../inference/tokenizers/behavior-flags.js';
import {
  normalizeQ4KLayout,
  resolveManifestQuantization,
  resolveEffectiveQuantizationInfo,
} from './quantization-info.js';
import {
  float16ToFloat32,
  float32ToFloat16,
  quantizeToQ4KM,
  quantizeToQ4KMRowWise,
  quantizeToQ4KMColumnWise,
  quantizeToInt4PerRowSymmetric,
} from './quantizer.js';
import { cloneJsonValue } from '../formats/clone-json.js';
import { SOURCE_PACKED_QUANT_DTYPES, isGemma4PerLayerEmbedTensor, normalizeModulesToNotConvert, normalizeStorageQuant, resolveQuantizeEmbeddings, transformTensorBytes } from './tensor-transform.js';
import { buildArtifactIdentity, extractArchitecture, getNestedTextConfig, sanitizeModelId } from './artifact-identity.js';
import { RDRR_VERSION, createManifest, modelHasMoETensors } from './manifest-builder.js';
import { normalizeCompressedTensorsW4A16, normalizeTensorName } from './tokenizer-contract.js';
export { RDRR_VERSION, buildBundledTokenizer, buildSentencepieceTokenizer, createManifest, inferEmbeddingOutputConfig, resolveBundledTokenizerVocabSize, resolveConvertedAt, resolveManifestMoEConfig, resolveManifestMultimodalConfig } from './manifest-builder.js';
export { extractArchitecture, sanitizeModelId } from './artifact-identity.js';
export { isGemma4PerLayerEmbedTensor, normalizeStorageQuant, resolveTensorTargetQuant, shouldQuantize, transformTensorBytes } from './tensor-transform.js';

// ============================================================================
// Re-exports for Backward Compatibility
// ============================================================================

export const ConvertStage = SchemaConversionStage;

// Re-export constants
export const SHARD_SIZE = SCHEMA_SHARD_SIZE;

// ============================================================================
// Embedding Output Inference
// ============================================================================

// ============================================================================
// Pure Functions (no I/O, no platform dependencies)
// ============================================================================

function shouldExcludeTextOnlyTensor(name) {
  const lower = name.toLowerCase();
  return lower.startsWith('vision_tower.')
    || lower.startsWith('model.vision_tower.')
    || lower.startsWith('model.encoder.vision_tower.')
    || lower.startsWith('vision_model.')
    || lower.startsWith('model.vision_model.')
    || lower.startsWith('model.encoder.vision_model.')
    || lower.startsWith('visual.')
    || lower.startsWith('model.visual.')
    || lower.startsWith('model.encoder.visual.')
    || lower.startsWith('embed_vision.')
    || lower.startsWith('model.embed_vision.')
    || lower.startsWith('model.encoder.embed_vision.')
    || lower.startsWith('vision.')
    || lower.startsWith('model.vision.')
    || lower.startsWith('model.encoder.vision.')
    || lower.startsWith('vision_encoder.')
    || lower.startsWith('model.encoder.vision_encoder.')
    || lower.startsWith('image_encoder.')
    || lower.startsWith('model.encoder.image_encoder.')
    || lower.startsWith('image_tower.')
    || lower.startsWith('model.encoder.image_tower.')
    || lower.startsWith('image.')
    || lower.startsWith('model.image.')
    || lower.startsWith('model.encoder.image.')
    || lower.startsWith('audio_tower.')
    || lower.startsWith('model.audio_tower.')
    || lower.startsWith('model.encoder.audio_tower.')
    || lower.startsWith('audio_model.')
    || lower.startsWith('model.audio_model.')
    || lower.startsWith('model.encoder.audio_model.')
    || lower.startsWith('audio.')
    || lower.startsWith('model.audio.')
    || lower.startsWith('model.encoder.audio.')
    || lower.startsWith('audio_encoder.')
    || lower.startsWith('model.encoder.audio_encoder.')
    || lower.startsWith('multi_modal_projector.')
    || lower.startsWith('model.multi_modal_projector.')
    || lower.startsWith('model.encoder.multi_modal_projector.')
    || lower.startsWith('mm_projector.')
    || lower.startsWith('model.mm_projector.')
    || lower.startsWith('model.encoder.mm_projector.');
}

function resolveConversionTensors(model, converterConfig) {
  const source = Array.isArray(model?.tensors) ? model.tensors : [];
  if (source.length === 0) {
    return source;
  }
  const textOnly = converterConfig?.output?.textOnly === true;
  if (!textOnly) {
    return source;
  }

  const hasLanguageModelNamespace = source.some((tensor) => {
    const lower = normalizeTensorName(tensor).toLowerCase();
    return lower.startsWith('language_model.') || lower.startsWith('model.language_model.');
  });
  if (hasLanguageModelNamespace) {
    return source.filter((tensor) => {
      const lower = normalizeTensorName(tensor).toLowerCase();
      // Keep top-level lm_head/output tensors alongside the language_model.*
      // namespace. Multimodal HF models (e.g. Qwen 3.6-27B) place the
      // language model body under model.language_model.* but expose the
      // language modeling head at the bare top level (`lm_head.weight`).
      // Dropping it leaves text-only conversion without an LM head and
      // pipeline init fails at loadWeights.
      return lower.startsWith('language_model.')
        || lower.startsWith('model.language_model.')
        || lower === 'lm_head.weight'
        || lower === 'model.lm_head.weight';
    });
  }

  return source.filter((tensor) => (
    !shouldExcludeTextOnlyTensor(normalizeTensorName(tensor))
  ));
}

function shouldMaterializeTiedLmHead(tensors, options) {
  if (options?.inference?.output?.tieWordEmbeddings !== true) {
    return false;
  }
  const lmHeadQuant = normalizeStorageQuant(options?.quantizationInfo?.lmHead ?? null);
  if (!SOURCE_PACKED_QUANT_DTYPES.has(lmHeadQuant) && lmHeadQuant !== 'q4k') {
    return false;
  }
  const embeddingQuant = normalizeStorageQuant(options?.quantizationInfo?.embeddings ?? null);
  if (embeddingQuant === lmHeadQuant) {
    return false;
  }
  return !tensors.some((tensor) => resolveTensorRole(tensor) === 'lm_head');
}

function resolveTiedLmHeadName(embeddingName) {
  const name = typeof embeddingName === 'string' ? embeddingName.trim() : '';
  if (name === 'model.language_model.model.embed_tokens.weight') {
    return 'model.language_model.lm_head.weight';
  }
  if (name === 'language_model.model.embed_tokens.weight') {
    return 'language_model.lm_head.weight';
  }
  if (name === 'model.language_model.embed_tokens.weight') {
    return 'model.language_model.lm_head.weight';
  }
  if (name === 'model.decoder.embed_tokens.weight') {
    return 'lm_head.weight';
  }
  if (name === 'decoder.embed_tokens.weight') {
    return 'lm_head.weight';
  }
  if (name === 'model.encoder.language_model.embed_tokens.weight') {
    return 'lm_head.weight';
  }
  if (name === 'language_model.embed_tokens.weight') {
    return 'language_model.lm_head.weight';
  }
  if (name === 'model.embed_tokens.weight' || name === 'embed_tokens.weight') {
    return 'lm_head.weight';
  }
  return 'lm_head.weight';
}

function resolveTiedEmbeddingTensor(tensors, modelType) {
  const candidates = tensors.filter((tensor) => (
    resolveTensorRole(tensor) === 'embedding'
    && resolveTensorGroup(tensor, modelType) === 'embed'
    && Array.isArray(tensor?.shape)
    && tensor.shape.length === 2
  ));
  return candidates[0] ?? null;
}

function materializeTiedLmHeadTensor(tensors, options) {
  if (!shouldMaterializeTiedLmHead(tensors, options)) {
    return tensors;
  }
  const embedding = resolveTiedEmbeddingTensor(tensors, options?.modelType ?? 'transformer');
  if (!embedding) {
    throw new Error(
      'Cannot materialize tied Q4K LM head: no 2D token embedding tensor was selected. '
      + 'Check inference.output.tieWordEmbeddings and the conversion tensor filter.'
    );
  }
  const name = resolveTiedLmHeadName(embedding.name);
  if (tensors.some((tensor) => tensor?.name === name)) {
    throw new Error(`Cannot materialize tied Q4K LM head: synthetic tensor name "${name}" already exists.`);
  }
  return [
    ...tensors,
    {
      ...embedding,
      name,
      role: 'lm_head',
      group: 'head',
      sourceTensorName: embedding.name,
    },
  ];
}

// Re-export formatBytes from schema for backward compatibility
export { formatBytes };

function validateInt4PleMaterializationContract(tensorLocations, inference, modelId) {
  const materialization = inference?.session?.perLayerInputs?.materialization;
  if (materialization !== 'gpu_split_tables') {
    return;
  }
  for (const [name, location] of Object.entries(tensorLocations ?? {})) {
    const sourceTransform = location?.sourceTransform ?? null;
    if (
      isGemma4PerLayerEmbedTensor(name)
      && sourceTransform?.kind === 'litert_axis_dequant'
      && String(sourceTransform?.sourceDtype ?? '').toUpperCase() === 'INT4'
    ) {
      throw new Error(
        `Manifest "${modelId}" Gemma 4 INT4 PLE tensor "${name}" cannot use ` +
        'inference.session.perLayerInputs.materialization="gpu_split_tables". ' +
        'Use materialization="range_backed" or disable INT4 PLE quantization.'
      );
    }
  }
}

function resolveGemma4TextConfig(rawConfig) {
  const textConfig = getNestedTextConfig(rawConfig);
  const modelType = String(textConfig?.model_type ?? rawConfig?.model_type ?? '').trim().toLowerCase();
  if (
    modelType !== 'gemma4'
    && modelType !== 'gemma4_text'
    && modelType !== 'gemma4_unified'
    && modelType !== 'gemma4_unified_text'
  ) {
    return null;
  }
  return textConfig ?? rawConfig ?? null;
}

function collectGemma4UnsupportedTensorFlags(tensors) {
  const names = Array.isArray(tensors) ? tensors.map((tensor) => normalizeTensorName(tensor).toLowerCase()) : [];
  const flags = [];
  if (names.some((name) => name.includes('.experts.gate_up_proj'))) {
    flags.push('experts.gate_up_proj');
  }
  if (names.some((name) => name.includes('.experts.down_proj'))) {
    flags.push('experts.down_proj');
  }
  if (names.some((name) => name.includes('.router.per_expert_scale'))) {
    flags.push('router.per_expert_scale');
  }
  if (names.some((name) => name.includes('.router.scale'))) {
    flags.push('router.scale');
  }
  if (names.some((name) => name.includes('.post_feedforward_layernorm_1.'))) {
    flags.push('post_feedforward_layernorm_1');
  }
  if (names.some((name) => name.includes('.post_feedforward_layernorm_2.'))) {
    flags.push('post_feedforward_layernorm_2');
  }
  if (names.some((name) => name.includes('.pre_feedforward_layernorm_2.'))) {
    flags.push('pre_feedforward_layernorm_2');
  }
  return flags;
}

function assertSupportedGemma4Conversion(model, tensors, modelId) {
  const rawConfig = model?.config ?? null;
  const textConfig = resolveGemma4TextConfig(rawConfig);
  if (!textConfig) return;

  const hiddenSizePerLayerInput = Number(textConfig.hidden_size_per_layer_input ?? 0);
  if (Number.isFinite(hiddenSizePerLayerInput) && hiddenSizePerLayerInput > 0) {
    const names = Array.isArray(tensors)
      ? tensors.map((tensor) => String(tensor?.name ?? '').trim())
      : [];
    const requiredNames = [
      'embed_tokens_per_layer.weight',
      'per_layer_input_gate.weight',
      'per_layer_projection.weight',
      'post_per_layer_input_norm.weight',
      'per_layer_model_projection.weight',
      'per_layer_projection_norm.weight',
    ];
    const missing = requiredNames.filter((suffix) => !names.some((name) => name.endsWith(suffix)));
    if (missing.length > 0) {
      throw new Error(
        `Gemma 4 model "${modelId}" declares hidden_size_per_layer_input=${hiddenSizePerLayerInput}, ` +
        `but the checkpoint is missing required per-layer input tensors: ${missing.join(', ')}.`
      );
    }
  }

  if (textConfig.enable_moe_block !== true) {
    return;
  }

  const unsupportedFlags = collectGemma4UnsupportedTensorFlags(tensors);
  if (unsupportedFlags.length === 0 && !modelHasMoETensors({ tensors })) {
    return;
  }

  throw new Error(
    `Gemma 4 model "${modelId}" is not supported yet: Gemma 4 MoE decoder blocks require ` +
    'Gemma-specific router scaling and dual dense+MoE FFN execution, but current Doppler MoE runtime ' +
    'only supports Mixtral/GPT-OSS semantics. ' +
    `Detected: ${unsupportedFlags.length > 0 ? unsupportedFlags.join(', ') : 'Gemma 4 MoE tensors'}.`
  );
}

export function buildTensorMap(tensors, shardSize) {
  if (!shardSize || shardSize <= 0) {
    throw new Error('Missing shard size for tensor map');
  }
  const tensorMap = {};

  let globalOffset = 0;
  for (const tensor of tensors) {
    const startShard = Math.floor(globalOffset / shardSize);
    const offsetInShard = globalOffset % shardSize;

    if (offsetInShard + tensor.size <= shardSize) {
      // Fits in single shard
      tensorMap[tensor.name] = {
        shard: startShard,
        offset: offsetInShard,
        size: tensor.size,
        shape: tensor.shape,
        dtype: tensor.dtype,
      };
    } else {
      // Spans multiple shards
      const spans = [];
      let remaining = tensor.size;
      let currentShard = startShard;
      let currentOffset = offsetInShard;

      while (remaining > 0) {
        const available = shardSize - currentOffset;
        const chunkSize = Math.min(remaining, available);
        spans.push({
          shardIndex: currentShard,
          offset: currentOffset,
          size: chunkSize,
        });
        remaining -= chunkSize;
        currentShard++;
        currentOffset = 0;
      }

      tensorMap[tensor.name] = {
        spans,
        size: tensor.size,
        shape: tensor.shape,
        dtype: tensor.dtype,
      };
    }

    globalOffset += tensor.size;
  }

  return tensorMap;
}

// ============================================================================
// Main Converter (uses I/O adapter)
// ============================================================================

const MAX_TENSOR_TYPED_ARRAY_BYTES = 0x7fff_ffff;

export async function convertModel(model, io, options = {}) {
  const { onProgress, signal } = options;
  const converterConfig = options.converterConfig || createConverterConfig();
  const shardSize = options.shardSize ?? converterConfig.sharding.shardSizeBytes;
  if (!shardSize || shardSize <= 0) {
    throw new Error('Missing shardSize for conversion');
  }
  const modelIdInput = (
    options.modelId
    ?? converterConfig.output.modelBaseId
    ?? model.modelId
    ?? model.name
  );
  const modelId = modelIdInput ? sanitizeModelId(modelIdInput) : null;
  if (!modelId) {
    throw new Error('Missing modelId for conversion');
  }
  const tensors = materializeTiedLmHeadTensor(
    normalizeCompressedTensorsW4A16(
      resolveConversionTensors(model, converterConfig),
      converterConfig
    ),
    {
      inference: options.inference ?? converterConfig?.inference ?? null,
      quantizationInfo: options.quantizationInfo ?? null,
      modelType: options.modelType ?? model.modelType ?? 'transformer',
    }
  );
  if (!Array.isArray(tensors) || tensors.length === 0) {
    const textOnly = converterConfig?.output?.textOnly === true;
    if (textOnly) {
      throw new Error(
        'No tensors selected for text-only conversion. ' +
        'Expected language_model.* tensors or non-vision tensor names.'
      );
    }
    throw new Error('Missing tensors for conversion');
  }
  assertSupportedGemma4Conversion(model, tensors, modelId);
  const totalTensors = tensors.length;
  const targetQuant = String(options.quantization ?? model.quantization ?? '').trim().toLowerCase();
  const tensorGroupModelType = String(options.modelType ?? model.modelType ?? 'transformer');
  const q4kLayout = normalizeQ4KLayout(options.quantizationInfo?.layout);
  const quantizeEmbeddings = resolveQuantizeEmbeddings(
    options.quantizationInfo ?? null,
    options.quantizeEmbeddings
  );
  const modulesToNotConvert = normalizeModulesToNotConvert(
    converterConfig?.quantization?.modulesToNotConvert ?? null
  );
  const shards = [];
  const tensorLocations = {};

  // Current shard accumulator
  let currentShardIndex = 0;
  let currentShardBuffer = new Uint8Array(shardSize);
  let currentShardSize = 0;
  let totalSize = 0;

  // Helper to flush current shard
  const flushShard = async () => {
    if (currentShardSize === 0) return;
    const shardData = currentShardBuffer.subarray(0, currentShardSize);

    // Write shard and get hash
    const hash = await io.writeShard(currentShardIndex, shardData);

    shards.push({
      index: currentShardIndex,
      filename: generateShardFilename(currentShardIndex),
      size: currentShardSize,
      hash,
      offset: currentShardIndex * shardSize,
    });

    currentShardIndex++;
    currentShardSize = 0;
  };

  const appendTensorBytes = async (tensorData, tensorSpans) => {
    if (!(tensorData instanceof Uint8Array)) {
      throw new Error('appendTensorBytes requires Uint8Array data.');
    }

    let remainingOffset = 0;
    while (remainingOffset < tensorData.length) {
      const availableInShard = shardSize - currentShardSize;
      const remainingSize = tensorData.length - remainingOffset;
      const chunkSize = Math.min(remainingSize, availableInShard);
      const chunk = tensorData.subarray(remainingOffset, remainingOffset + chunkSize);
      currentShardBuffer.set(chunk, currentShardSize);

      const chunkOffset = currentShardSize;
      currentShardSize += chunkSize;
      totalSize += chunkSize;

      tensorSpans.push({
        shardIndex: currentShardIndex,
        offset: chunkOffset,
        size: chunkSize,
      });

      remainingOffset += chunkSize;

      if (currentShardSize >= shardSize) {
        await flushShard();
      }
    }
  };

  // Process tensors
  for (let i = 0; i < tensors.length; i++) {
    if (signal?.aborted) {
      throw new DOMException('Conversion cancelled', 'AbortError');
    }

    const tensor = tensors[i];

    onProgress?.({
      stage: ConvertStage.WRITING,
      message: `Processing ${tensor.name}`,
      current: i + 1,
      total: totalTensors,
      percent: Math.round(((i + 1) / totalTensors) * 100),
    });

    const transformContext = {
      targetQuant,
      q4kLayout,
      quantizationInfo: options.quantizationInfo ?? null,
      quantizeEmbeddings,
      modulesToNotConvert,
    };
    const reportTensorProgress = (currentBytes, totalBytes) => {
      if (!Number.isFinite(currentBytes) || !Number.isFinite(totalBytes)) return;
      onProgress?.({
        stage: ConvertStage.WRITING,
        message: `Processing ${tensor.name}`,
        current: i + 1,
        total: totalTensors,
        percent: Math.round(((i + 1) / totalTensors) * 100),
        tensorName: tensor.name,
        tensorBytesCurrent: currentBytes,
        tensorBytesTotal: totalBytes,
      });
    };
    const tensorSpans = [];
    const sourceTensorSize = Number.isFinite(tensor?.size) ? Number(tensor.size) : null;
    let outDtype = tensor.dtype;
    let outLayout = null;
    let tensorStorage = null;
    let tensorSize = 0;

    if (
      sourceTensorSize != null
      && sourceTensorSize > MAX_TENSOR_TYPED_ARRAY_BYTES
    ) {
      if (typeof options.largeTensorTransformer !== 'function') {
        throw new Error(
          `Tensor "${tensor.name}" is ${formatBytes(sourceTensorSize)} and exceeds the single-buffer conversion limit ` +
          `(${formatBytes(MAX_TENSOR_TYPED_ARRAY_BYTES)}). Provide a largeTensorTransformer for streamed conversion.`
        );
      }

      let emittedChunk = false;
      // For PLE INT4 per-row quantization, each row-chunk returns its own
      // per-row F32 scale slice via companionData. Accumulate them across
      // chunks; after the stream completes, write the concatenated scales
      // blob and attach sourceTransform.scaleSource pointing at it.
      const companionChunks = [];
      let accumulatedSourceTransform = null;
      await options.largeTensorTransformer({
        tensor,
        transformContext,
        reportProgress: reportTensorProgress,
        async writeChunk(result) {
          const tensorData = result?.tensorData;
          if (!(tensorData instanceof Uint8Array)) {
            throw new Error(`Large tensor transformer must return Uint8Array data for ${tensor.name}.`);
          }
          const chunkOutDtype = result?.outDtype ?? tensor.dtype;
          const chunkOutLayout = result?.outLayout ?? null;
          const chunkStorage = result?.storage ?? null;
          if (!emittedChunk) {
            outDtype = chunkOutDtype;
            outLayout = chunkOutLayout;
            tensorStorage = chunkStorage;
            emittedChunk = true;
          } else {
            if (chunkOutDtype !== outDtype) {
              throw new Error(`Large tensor transformer returned inconsistent dtype for ${tensor.name}.`);
            }
            if (chunkOutLayout !== outLayout) {
              throw new Error(`Large tensor transformer returned inconsistent layout for ${tensor.name}.`);
            }
            if (JSON.stringify(chunkStorage) !== JSON.stringify(tensorStorage)) {
              throw new Error(`Large tensor transformer returned inconsistent storage descriptor for ${tensor.name}.`);
            }
          }
          tensorSize += tensorData.byteLength;
          await appendTensorBytes(tensorData, tensorSpans);

          if (result?.companionData instanceof Uint8Array && result.companionData.byteLength > 0) {
            if (!result.sourceTransform) {
              throw new Error(
                `Large tensor chunk returned companionData without sourceTransform for ${tensor.name}.`
              );
            }
            companionChunks.push(result.companionData);
            if (!accumulatedSourceTransform) {
              accumulatedSourceTransform = { ...result.sourceTransform };
            }
          } else if (result?.sourceTransform && !accumulatedSourceTransform) {
            accumulatedSourceTransform = { ...result.sourceTransform };
          }
        },
      });

      if (!emittedChunk) {
        throw new Error(`Large tensor transformer did not emit any bytes for ${tensor.name}.`);
      }

      if (accumulatedSourceTransform) {
        let companionBytes = null;
        if (companionChunks.length > 0) {
          let totalCompanionBytes = 0;
          for (const chunk of companionChunks) totalCompanionBytes += chunk.byteLength;
          companionBytes = new Uint8Array(totalCompanionBytes);
          let offset = 0;
          for (const chunk of companionChunks) {
            companionBytes.set(chunk, offset);
            offset += chunk.byteLength;
          }
          const companionSpans = [];
          await appendTensorBytes(companionBytes, companionSpans);
          if (companionSpans.length !== 1) {
            throw new Error(
              `Companion scales for ${tensor.name} must land in a single shard (got ${companionSpans.length}).`
            );
          }
          transformContext._pleSourceTransform = {
            ...accumulatedSourceTransform,
            scaleSource: {
              shard: companionSpans[0].shardIndex,
              offset: companionSpans[0].offset,
              size: companionBytes.byteLength,
            },
          };
        } else {
          transformContext._pleSourceTransform = accumulatedSourceTransform;
        }
      }
    } else {
      const data = await io.readTensorData(tensor);
      const tensorDataInput = new Uint8Array(data);
      const transformResult = (
        typeof options.tensorTransformer === 'function'
          ? await options.tensorTransformer({
            tensor,
            tensorData: tensorDataInput,
            transformContext,
            reportProgress: reportTensorProgress,
          })
          : transformTensorBytes(tensor, tensorDataInput, transformContext)
      );

      const tensorData = transformResult?.tensorData;
      if (!(tensorData instanceof Uint8Array)) {
        throw new Error(`Tensor transformer must return Uint8Array data for ${tensor.name}.`);
      }
      outDtype = transformResult?.outDtype ?? tensor.dtype;
      outLayout = transformResult?.outLayout ?? null;
      tensorStorage = transformResult?.storage ?? null;
      tensorSize = tensorData.byteLength;
      await appendTensorBytes(tensorData, tensorSpans);

      // Companion data (e.g., INT4 per-row scales for PLE) writes to its own
      // spans and is referenced from sourceTransform.scaleSource on the
      // primary tensor location.
      const companionData = transformResult?.companionData;
      if (companionData instanceof Uint8Array && companionData.byteLength > 0) {
        if (!transformResult?.sourceTransform) {
          throw new Error(
            `Tensor transformer returned companionData without sourceTransform for ${tensor.name}.`
          );
        }
        const companionSpans = [];
        await appendTensorBytes(companionData, companionSpans);
        if (companionSpans.length !== 1) {
          throw new Error(
            `Companion scales for ${tensor.name} must land in a single shard (got ${companionSpans.length}).`
          );
        }
        transformContext._pleSourceTransform = {
          ...transformResult.sourceTransform,
          scaleSource: {
            shard: companionSpans[0].shardIndex,
            offset: companionSpans[0].offset,
            size: companionData.byteLength,
          },
        };
      } else if (transformResult?.sourceTransform) {
        transformContext._pleSourceTransform = transformResult.sourceTransform;
      } else {
        transformContext._pleSourceTransform = null;
      }
    }

    // Record tensor location
    const role = resolveTensorRole(tensor);
    const group = resolveTensorGroup(tensor, tensorGroupModelType);
    const pleSourceTransform = transformContext._pleSourceTransform ?? null;
    transformContext._pleSourceTransform = null;

    if (tensorSpans.length === 1) {
      tensorLocations[tensor.name] = {
        shard: tensorSpans[0].shardIndex,
        offset: tensorSpans[0].offset,
        size: tensorSize,
        shape: tensor.shape,
        dtype: outDtype,
        role,
        group,
        ...(outLayout ? { layout: outLayout } : {}),
        ...(tensorStorage ? { storage: tensorStorage } : {}),
        ...(pleSourceTransform ? { sourceTransform: pleSourceTransform } : {}),
      };
    } else {
      tensorLocations[tensor.name] = {
        spans: tensorSpans,
        size: tensorSize,
        shape: tensor.shape,
        dtype: outDtype,
        role,
        group,
        ...(outLayout ? { layout: outLayout } : {}),
        ...(tensorStorage ? { storage: tensorStorage } : {}),
        ...(pleSourceTransform ? { sourceTransform: pleSourceTransform } : {}),
      };
    }

  }

  // Flush final shard
  await flushShard();

  if (signal?.aborted) {
    throw new DOMException('Conversion cancelled', 'AbortError');
  }

  // Create manifest
  onProgress?.({
    stage: ConvertStage.MANIFEST,
    message: 'Creating manifest...',
  });

  const tensorEntries = Object.entries(tensorLocations).map(([name, location]) => ({
    name,
    dtype: location?.dtype ?? null,
    role: location?.role ?? null,
    layout: location?.layout ?? null,
  }));
  const effectiveQuantizationInfo = resolveEffectiveQuantizationInfo(
    options.quantizationInfo ?? null,
    tensorEntries
  );
  const effectiveManifestQuantization = resolveManifestQuantization(
    effectiveQuantizationInfo.weights,
    options.quantization ?? model.quantization
  );

  validateInt4PleMaterializationContract(tensorLocations, options.inference, modelId);

  const artifactIdentity = await buildArtifactIdentity({
    modelId,
    modelType: options.modelType,
    source: options.source,
    sourcePath: options.sourcePath,
    sourceFormat: options.sourceFormat,
    conversionConfigPath: options.conversionConfigPath,
    conversionConfig: options.conversionConfig,
    explicitArtifactIdentity: converterConfig?.manifest?.artifactIdentity ?? options.artifactIdentity ?? null,
    hashString: options.hashString,
    hashAlgorithm: converterConfig.manifest.hashAlgorithm,
    shards,
    tensorLocations,
    quantizationInfo: effectiveQuantizationInfo,
    inference: options.inference,
    manifestConfig: converterConfig?.manifest ?? null,
    converterConfig,
  });

  const manifest = createManifest(modelId, model, shards, tensorLocations, {
    source: options.source ?? 'convert-core',
    modelType: options.modelType,
    quantization: effectiveManifestQuantization,
    quantizationInfo: effectiveQuantizationInfo,
    moeConfig: converterConfig?.moeConfig ?? options.moeConfig ?? null,
    artifactIdentity,
    weightsRef: converterConfig?.manifest?.weightsRef ?? options.weightsRef ?? null,
    hashAlgorithm: converterConfig.manifest.hashAlgorithm,
    architecture: options.architecture,
    inference: options.inference,
    eosTokenId: converterConfig?.manifest?.eosTokenId ?? options.eosTokenId,
    convertedAt: converterConfig?.manifest?.conversion?.convertedAt ?? null,
    conversionInfo: converterConfig?.manifest?.conversion ?? null,
    manifestConfig: converterConfig?.manifest ?? null,
    textOnly: converterConfig?.output?.textOnly === true,
  });

  // Write manifest
  await io.writeManifest(manifest);

  onProgress?.({
    stage: ConvertStage.COMPLETE,
    message: 'Conversion complete!',
    modelId,
    shardCount: shards.length,
    totalSize: formatBytes(totalSize),
  });

  const executionContractArtifact = buildExecutionContractArtifact(manifest);
  const layerPatternContractArtifact = getInferenceLayerPatternContractArtifact();
  const requiredInferenceFieldsArtifact = manifest?.modelType === 'transformer'
    && manifest?.inference
    && typeof manifest.inference === 'object'
    && manifest.inference.attention
    && typeof manifest.inference.attention === 'object'
    ? buildManifestRequiredInferenceFieldsArtifact(
      manifest?.inference ?? null,
      `${manifest?.modelId ?? modelId}.inference`
    )
    : null;
  return {
    manifest,
    shardCount: shards.length,
    tensorCount: tensors.length,
    totalSize,
    executionContractArtifact,
    layerPatternContractArtifact,
    requiredInferenceFieldsArtifact,
  };
}

// ============================================================================
// Utility Exports
// ============================================================================

export { generateShardFilename };
