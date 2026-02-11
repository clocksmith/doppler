

import { parseGGUFHeaderFromSource } from './gguf-parser-browser.js';
import { isTensorSource, normalizeTensorSource } from './tensor-source-file.js';
import { createRemoteTensorSource } from './tensor-source-download.js';
import {
  parseSafetensorsFile,
  parseSafetensorsSharded,
  parseConfigJson,
  parseTokenizerConfigJson,
  parseTokenizerJson,
  parseIndexJson,
  streamTensorData,
  detectModelFormat,
  getAuxiliaryFiles,
} from './safetensors-parser-browser.js';
import {
  initStorage,
  openModelStore,
  saveManifest,
  saveTokenizer,
  saveTokenizerModel,
  saveAuxFile,
  deleteModel,
  createConversionShardWriter,
  computeHash,
  createStreamingHasher,
  getStorageBackendType,
} from '../storage/shard-manager.js';
import { registerModel } from '../storage/registry.js';
import {
  checkSpaceAvailable,
  requestPersistence,
  isOPFSAvailable,
  isIndexedDBAvailable,
  QuotaExceededError,
} from '../storage/quota.js';

// Import shared shard packing logic
import {
  ShardPacker,
} from '../converter/shard-packer.js';
import { classifyTensorRole } from '../storage/rdrr-format.js';
import {
  buildQuantizationInfo,
  resolveManifestQuantization,
  resolveModelId,
  resolveTensorDtype,
  resolveQ4KLayout,
  getQ4KOutputSize,
  createQ4KChunkStream,
  createF16ChunkStream,
} from './quantization.js';

// Import shared types and functions from convert-core
import {
  ConvertStage,
  sanitizeModelId,
  formatBytes,
  extractArchitecture,
  createManifest,
} from '../converter/core.js';
import { buildManifestInference } from '../converter/manifest-inference.js';

import { createConverterConfig, detectPreset, resolvePreset } from '../config/index.js';
import { DEFAULT_MANIFEST_INFERENCE } from '../config/schema/index.js';
import { MB, GB } from '../config/schema/units.schema.js';
import { log, trace } from '../debug/index.js';

// Re-export types for consumers
export {
  ConvertStage,
};

export function isConversionSupported() {
  return isOPFSAvailable() || isIndexedDBAvailable();
}

const LARGE_MODEL_THRESHOLD_BYTES = 8 * GB;
const LARGE_MODEL_CHUNK_BYTES = 128 * MB;
const LARGE_MODEL_SHARD_BYTES = 256 * MB;

function tuneConverterConfig(config, totalInputBytes, modelType) {
  if (!config) return;
  const isLarge = Number.isFinite(totalInputBytes) && totalInputBytes >= LARGE_MODEL_THRESHOLD_BYTES;
  const isDiffusion = modelType === 'diffusion';
  if (!isLarge && !isDiffusion) return;

  if (config.streaming?.chunkSizeBytes) {
    config.streaming.chunkSizeBytes = Math.max(config.streaming.chunkSizeBytes, LARGE_MODEL_CHUNK_BYTES);
  }
  if (config.sharding?.shardSizeBytes) {
    config.sharding.shardSizeBytes = Math.max(config.sharding.shardSizeBytes, LARGE_MODEL_SHARD_BYTES);
  }
}

function createStageTimer(label) {
  const start = performance.now();
  return {
    stop: (extra, data) => {
      const elapsed = performance.now() - start;
      const suffix = extra ? ` - ${extra}` : '';
      log.info('Convert', `${label}: ${elapsed.toFixed(0)}ms${suffix}`);
      trace.perf(`Convert ${label}: ${elapsed.toFixed(0)}ms`, data);
      return elapsed;
    },
  };
}

function resolveRemoteOptions(options) {
  const http = options?.converterConfig?.http || null;
  return {
    headers: options?.headers,
    signal: options?.signal,
    name: options?.name,
    allowDownloadFallback: http?.allowDownloadFallback,
    maxDownloadBytes: http?.maxDownloadBytes,
  };
}

export async function createRemoteModelSources(urls, options = {}) {
  if (!Array.isArray(urls) || urls.length === 0) {
    throw new Error('Remote conversion requires at least one URL.');
  }

  const sources = [];
  const remoteOptions = resolveRemoteOptions(options);
  for (const url of urls) {
    if (typeof url !== 'string' || url.length === 0) {
      throw new Error('Remote conversion URLs must be non-empty strings.');
    }
    const result = await createRemoteTensorSource(url, remoteOptions);
    sources.push(result.source);
  }

  return sources;
}

// ============================================================================
// Main Convert Function
// ============================================================================


function normalizeWeightDtype(dtype) {
  const upper = String(dtype || '').toUpperCase();
  return upper === 'BF16' ? 'F16' : upper;
}

function inferQuantizationFromTensors(tensors) {
  const weightDtypes = new Set();
  for (const tensor of tensors) {
    if (!tensor?.name || typeof tensor.dtype !== 'string') continue;
    if (!tensor.name.includes('.weight')) continue;
    weightDtypes.add(normalizeWeightDtype(tensor.dtype));
  }
  if (weightDtypes.size === 0) return null;
  if (weightDtypes.size > 1) {
    throw new Error(`Ambiguous weight dtypes: ${Array.from(weightDtypes).join(', ')}`);
  }
  return Array.from(weightDtypes)[0];
}

function isEmbeddingTensorName(name) {
  return classifyTensorRole(name) === 'embedding';
}

function isLmHeadTensorName(name) {
  return classifyTensorRole(name) === 'lm_head';
}

function findTensorDtype(tensors, matcher) {
  const match = tensors.find((t) => matcher(t.name));
  return match?.dtype ?? null;
}

function getFilePath(file) {
  if (!file) return '';
  if (typeof file.relativePath === 'string' && file.relativePath.length > 0) {
    return file.relativePath;
  }
  if (typeof file.webkitRelativePath === 'string' && file.webkitRelativePath.length > 0) {
    return file.webkitRelativePath;
  }
  if (typeof file.name === 'string') return file.name;
  return '';
}

function normalizePath(path) {
  return String(path || '').replace(/\\/g, '/');
}

function getBaseName(path) {
  const normalized = normalizePath(path);
  if (!normalized) return '';
  const parts = normalized.split('/');
  return parts[parts.length - 1] || '';
}

function pathEndsWith(path, suffix) {
  const normalized = normalizePath(path);
  return normalized.endsWith(suffix);
}

function findFileBySuffix(files, suffix) {
  return files.find((file) => pathEndsWith(getFilePath(file), suffix)) || null;
}

async function readTextFile(file, label = 'file') {
  if (!file || typeof file.text !== 'function') {
    throw new Error(`Missing ${label}`);
  }
  return file.text();
}

async function parseJsonFile(file, label = 'json') {
  const text = await readTextFile(file, label);
  try {
    return JSON.parse(text);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`[${label}] Failed to parse JSON: ${message}`);
  }
}

function isDiffusionInput(files) {
  return files.some((file) => getBaseName(getFilePath(file)) === 'model_index.json');
}

function pickFirstBySuffix(files, suffixes) {
  for (const suffix of suffixes) {
    const match = findFileBySuffix(files, suffix);
    if (match) return match;
  }
  return null;
}

async function parseDiffusionModel(files, onProgress, signal) {
  const modelIndexFile = findFileBySuffix(files, 'model_index.json');
  if (!modelIndexFile) return null;

  onProgress?.({
    stage: ConvertStage.PARSING,
    message: 'Parsing diffusion model_index.json...',
  });

  const modelIndex = await parseJsonFile(modelIndexFile, 'model_index.json');
  const diffusionConfig = {
    modelIndex,
    components: {},
  };
  const auxFiles = [];
  const tensors = [];

  const addPrefixedTensors = (componentId, parsed) => {
    for (const tensor of parsed.tensors) {
      tensors.push({
        ...tensor,
        name: `${componentId}.${tensor.name}`,
      });
    }
  };

  const parseComponentConfig = async (componentId, suffix) => {
    const file = findFileBySuffix(files, suffix);
    if (!file) return null;
    const config = await parseJsonFile(file, `${componentId} config`);
    if (componentId === 'transformer' && config && !config.weight_format) {
      config.weight_format = 'diffusers';
    }
    diffusionConfig.components[componentId] = {
      ...(diffusionConfig.components[componentId] || {}),
      config,
    };
    return config;
  };

  const parseSingleSafetensors = async (componentId, file) => {
    if (!file) {
      throw new Error(`Missing ${componentId} safetensors file`);
    }
    const parsed = await parseSafetensorsFile(file);
    addPrefixedTensors(componentId, parsed);
  };

  const parseShardedSafetensors = async (componentId, indexFile) => {
    if (!indexFile) {
      throw new Error(`Missing ${componentId} sharded index file`);
    }
    const indexJson = await parseJsonFile(indexFile, `${componentId} index`);
    const weightMap = indexJson?.weight_map || {};
    const shardNames = Array.from(new Set(Object.values(weightMap)));
    if (shardNames.length === 0) {
      throw new Error(`No shards listed in ${componentId} index file`);
    }
    const indexPath = normalizePath(getFilePath(indexFile));
    const baseDir = indexPath.includes('/') ? indexPath.split('/').slice(0, -1).join('/') : '';
    const shardFiles = shardNames.map((name) => {
      const suffix = baseDir ? `${baseDir}/${name}` : name;
      return findFileBySuffix(files, suffix);
    }).filter(Boolean);
    if (shardFiles.length !== shardNames.length) {
      throw new Error(`Missing shard files for ${componentId} (${shardFiles.length}/${shardNames.length} found)`);
    }
    const parsed = await parseSafetensorsSharded(shardFiles, indexJson);
    addPrefixedTensors(componentId, parsed);
  };

  if (signal?.aborted) throw new DOMException('Cancelled', 'AbortError');

  await parseComponentConfig('transformer', 'transformer/config.json');
  await parseComponentConfig('text_encoder', 'text_encoder/config.json');
  await parseComponentConfig('text_encoder_2', 'text_encoder_2/config.json');
  await parseComponentConfig('text_encoder_3', 'text_encoder_3/config.json');
  await parseComponentConfig('vae', 'vae/config.json');
  await parseComponentConfig('scheduler', 'scheduler/scheduler_config.json');

  onProgress?.({
    stage: ConvertStage.PARSING,
    message: 'Parsing diffusion weights...',
  });

  const transformerFile = pickFirstBySuffix(files, [
    'transformer/diffusion_pytorch_model.safetensors',
    'transformer/model.safetensors',
    'transformer/model.fp16.safetensors',
    'sd3.5_medium.safetensors',
  ]);
  await parseSingleSafetensors('transformer', transformerFile);

  const textEncoderFile = pickFirstBySuffix(files, [
    'text_encoder/model.safetensors',
    'text_encoder/model.fp16.safetensors',
  ]);
  await parseSingleSafetensors('text_encoder', textEncoderFile);

  const textEncoder2File = pickFirstBySuffix(files, [
    'text_encoder_2/model.safetensors',
    'text_encoder_2/model.fp16.safetensors',
  ]);
  await parseSingleSafetensors('text_encoder_2', textEncoder2File);

  const textEncoder3Index = pickFirstBySuffix(files, [
    'text_encoder_3/model.safetensors.index.json',
    'text_encoder_3/model.safetensors.index.fp16.json',
  ]);
  await parseShardedSafetensors('text_encoder_3', textEncoder3Index);

  const vaeFile = pickFirstBySuffix(files, [
    'vae/diffusion_pytorch_model.safetensors',
    'vae/model.safetensors',
  ]);
  await parseSingleSafetensors('vae', vaeFile);

  const storeTextAsset = async (label, suffix, targetName) => {
    const file = findFileBySuffix(files, suffix);
    if (!file) return;
    const text = await readTextFile(file, label);
    auxFiles.push({ name: targetName, data: text });
  };

  const storeBinaryAsset = async (label, suffix, targetName) => {
    const file = findFileBySuffix(files, suffix);
    if (!file) return;
    if (typeof file.arrayBuffer !== 'function') {
      throw new Error(`Missing binary data for ${label}`);
    }
    const buffer = await file.arrayBuffer();
    auxFiles.push({ name: targetName, data: buffer });
  };

  await storeTextAsset('tokenizer vocab', 'tokenizer/vocab.json', 'tokenizer_vocab.json');
  await storeTextAsset('tokenizer merges', 'tokenizer/merges.txt', 'tokenizer_merges.txt');
  await storeTextAsset('tokenizer config', 'tokenizer/tokenizer_config.json', 'tokenizer_config.json');
  await storeTextAsset('tokenizer special tokens', 'tokenizer/special_tokens_map.json', 'tokenizer_special_tokens_map.json');

  await storeTextAsset('tokenizer_2 vocab', 'tokenizer_2/vocab.json', 'tokenizer_2_vocab.json');
  await storeTextAsset('tokenizer_2 merges', 'tokenizer_2/merges.txt', 'tokenizer_2_merges.txt');
  await storeTextAsset('tokenizer_2 config', 'tokenizer_2/tokenizer_config.json', 'tokenizer_2_config.json');
  await storeTextAsset('tokenizer_2 special tokens', 'tokenizer_2/special_tokens_map.json', 'tokenizer_2_special_tokens_map.json');

  await storeTextAsset('tokenizer_3 json', 'tokenizer_3/tokenizer.json', 'tokenizer_3_tokenizer.json');
  await storeBinaryAsset('tokenizer_3 spiece', 'tokenizer_3/spiece.model', 'tokenizer_3_spiece.model');
  await storeTextAsset('tokenizer_3 config', 'tokenizer_3/tokenizer_config.json', 'tokenizer_3_config.json');
  await storeTextAsset('tokenizer_3 special tokens', 'tokenizer_3/special_tokens_map.json', 'tokenizer_3_special_tokens_map.json');

  const requireTokenizerAsset = (label, suffix) => {
    if (!findFileBySuffix(files, suffix)) {
      throw new Error(`Missing ${label} (${suffix}) for diffusion conversion.`);
    }
  };

  if (modelIndex?.tokenizer) {
    requireTokenizerAsset('tokenizer vocab', 'tokenizer/vocab.json');
    requireTokenizerAsset('tokenizer merges', 'tokenizer/merges.txt');
  }
  if (modelIndex?.tokenizer_2) {
    requireTokenizerAsset('tokenizer_2 vocab', 'tokenizer_2/vocab.json');
    requireTokenizerAsset('tokenizer_2 merges', 'tokenizer_2/merges.txt');
  }
  if (modelIndex?.tokenizer_3) {
    requireTokenizerAsset('tokenizer_3 json', 'tokenizer_3/tokenizer.json');
    requireTokenizerAsset('tokenizer_3 spiece', 'tokenizer_3/spiece.model');
  }

  diffusionConfig.tokenizers = {
    text_encoder: {
      type: 'bpe',
      vocabFile: 'tokenizer_vocab.json',
      mergesFile: 'tokenizer_merges.txt',
      configFile: 'tokenizer_config.json',
      specialTokensFile: 'tokenizer_special_tokens_map.json',
    },
    text_encoder_2: {
      type: 'bpe',
      vocabFile: 'tokenizer_2_vocab.json',
      mergesFile: 'tokenizer_2_merges.txt',
      configFile: 'tokenizer_2_config.json',
      specialTokensFile: 'tokenizer_2_special_tokens_map.json',
    },
    text_encoder_3: {
      type: 'sentencepiece',
      tokenizerFile: 'tokenizer_3_tokenizer.json',
      spieceFile: 'tokenizer_3_spiece.model',
      configFile: 'tokenizer_3_config.json',
      specialTokensFile: 'tokenizer_3_special_tokens_map.json',
    },
  };

  return {
    tensors,
    config: { diffusion: diffusionConfig },
    auxFiles,
    architecture: 'diffusion',
  };
}

export async function convertModel(files, options = {}) {
  const { modelId: userModelId, onProgress, signal, converterConfig } = options;
  const resolvedConverterConfig = converterConfig || createConverterConfig();
  const progressState = {
    lastStage: null,
    lastMessage: null,
    lastPercentBucket: null,
  };
  const reportProgress = (update) => {
    if (update) {
      const stage = update.stage ?? null;
      const message = update.message ? String(update.message) : '';
      const percent = Number.isFinite(update.percent) ? update.percent : null;
      if (stage && stage !== progressState.lastStage) {
        progressState.lastStage = stage;
        progressState.lastMessage = null;
        progressState.lastPercentBucket = null;
        log.info('Convert', message ? `${stage}: ${message}` : `${stage}`);
      } else if (stage === ConvertStage.WRITING && percent !== null) {
        const bucket = Math.floor(percent / 5) * 5;
        if (bucket !== progressState.lastPercentBucket) {
          progressState.lastPercentBucket = bucket;
          const counts = Number.isFinite(update.current) && Number.isFinite(update.total)
            ? ` (${update.current}/${update.total})`
            : '';
          log.info('Convert', `Writing ${bucket}%${counts}`);
        }
      } else if (message && message !== progressState.lastMessage) {
        progressState.lastMessage = message;
        log.info('Convert', message);
      }
    }
    onProgress?.(update);
  };

  let modelId = null;
  const shardInfos = [];
  const cleanupTasks = [];
  const inputFiles = Array.isArray(files) ? files : [];
  const totalInputBytes = inputFiles.reduce((sum, file) => sum + (file?.size || 0), 0);
  const hasOnlyFiles = inputFiles.every((file) => !isTensorSource(file));
  const diffusionCandidate = hasOnlyFiles && isDiffusionInput(inputFiles);
  let diffusionInfo = null;
  let diffusionAuxFiles = null;
  let diffusionArchitecture = null;
  let diffusionEosTokenId = undefined;
  const conversionStart = performance.now();
  log.info(
    'Convert',
    `Start: ${inputFiles.length} files, ${formatBytes(totalInputBytes)}`
  );

  try {
    if (!isOPFSAvailable() && !isIndexedDBAvailable()) {
      throw new Error('No supported storage backend available for browser conversion. Supported: opfs, indexeddb.');
    }

    // Initialize storage
    await initStorage();
    const persistence = await requestPersistence();
    const backendType = getStorageBackendType();
    reportProgress({
      stage: ConvertStage.DETECTING,
      message: `Storage backend: ${backendType ?? 'unknown'}`,
      backend: backendType,
      persistence,
    });

    // Detect format
    reportProgress({
      stage: ConvertStage.DETECTING,
      message: 'Detecting model format...',
    });

    if (signal?.aborted) throw new DOMException('Cancelled', 'AbortError');

    const detectTimer = createStageTimer('Detect format');
    if (diffusionCandidate) {
      diffusionInfo = await parseDiffusionModel(inputFiles, reportProgress, signal);
      diffusionAuxFiles = diffusionInfo?.auxFiles ?? null;
      diffusionArchitecture = diffusionInfo?.architecture ?? null;
      diffusionEosTokenId = null;
    }

    const format = diffusionInfo ? { type: 'diffusion' } : detectModelFormat(files);
    detectTimer.stop(`type=${format.type}`);
    const auxiliary = diffusionInfo ? null : getAuxiliaryFiles(files);
    for (const file of files) {
      if (isTensorSource(file) && typeof file.cleanup === 'function') {
        cleanupTasks.push(file.cleanup);
      }
    }
    if (!diffusionInfo) {
      const hasTokenizerJson = !!auxiliary.tokenizer;
      const hasTokenizerModel = !!auxiliary.tokenizerModel;
      if (!hasTokenizerJson && !hasTokenizerModel) {
        throw new Error('Missing tokenizer.json or tokenizer.model for browser conversion.');
      }
    }

    reportProgress({
      stage: ConvertStage.DETECTING,
      message: `Format: ${format.type}`,
      format: format.type,
    });

    // Parse based on format
    let modelInfo;
    let config = null;
    let tokenizerJson = null;
    let tokenizerConfig = null;
    let tokenizerModel = null;

    const parseTimer = createStageTimer('Parse tensors');
    if (format.type === 'diffusion') {
      modelInfo = {
        tensors: diffusionInfo.tensors,
        config: diffusionInfo.config,
        architecture: diffusionArchitecture,
        format: 'safetensors',
      };
      config = diffusionInfo.config;
    } else if (format.type === 'gguf') {
      modelInfo = await parseGGUFModel(format.ggufFile, reportProgress, signal);
    } else if (format.type === 'single') {
      const parsed = await parseSafetensorsFile(format.safetensorsFile);
      modelInfo = { tensors: parsed.tensors, config: parsed.config };
      if (auxiliary.config) {
        config = await parseConfigJson(auxiliary.config);
        modelInfo.config = config;
      }
    } else if (format.type === 'sharded' || format.type === 'sharded-no-index') {
      let indexJson = null;
      if (format.indexFile) {
        indexJson = await parseIndexJson(format.indexFile);
      }
      const parsed = await parseSafetensorsSharded(format.safetensorsFiles, indexJson);
      modelInfo = { tensors: parsed.tensors, config: parsed.config };
      if (auxiliary.config) {
        config = await parseConfigJson(auxiliary.config);
        modelInfo.config = config;
      }
    } else {
      throw new Error(`Unsupported format: ${format.type}`);
    }

    // Parse tokenizer if available (text models only)
    if (!diffusionInfo) {
      if (auxiliary.tokenizer) {
        tokenizerJson = await parseTokenizerJson(auxiliary.tokenizer);
        modelInfo.tokenizerJson = tokenizerJson;
      }
      if (auxiliary.tokenizerConfig) {
        tokenizerConfig = await parseTokenizerConfigJson(auxiliary.tokenizerConfig);
        modelInfo.tokenizerConfig = tokenizerConfig;
      }
      if (auxiliary.tokenizerModel) {
        const source = normalizeTensorSource(auxiliary.tokenizerModel);
        tokenizerModel = await source.readRange(0, source.size);
        modelInfo.tokenizerModel = tokenizerModel;
      }
    }
    parseTimer.stop(`${modelInfo.tensors.length} tensors`);

    if (signal?.aborted) throw new DOMException('Cancelled', 'AbortError');

    // Detect model type using preset system (skip for diffusion)
    const rawConfig = (config || modelInfo.config || {});
    let preset = null;
    let modelType = null;
    let manifestInference = null;
    let headDim = null;
    let tensorNames = null;

    if (diffusionInfo) {
      modelType = 'diffusion';
      manifestInference = { ...DEFAULT_MANIFEST_INFERENCE, presetId: 'diffusion' };
    } else {
      const presetOverride = resolvedConverterConfig.presets?.model;
      const presetId = presetOverride || detectPreset(rawConfig, modelInfo.architecture);
      if (presetId === 'transformer') {
        const modelType = rawConfig.model_type ?? 'unknown';
        throw new Error(
          `Unknown model family: architecture="${modelInfo.architecture || 'unknown'}", model_type="${modelType}"\n\n` +
          `DOPPLER requires a known model preset to generate correct inference config.\n` +
          `The manifest-first architecture does not support generic defaults.\n\n` +
          `Options:\n` +
          `  1. Wait for official support of this model family\n` +
          `  2. Create a custom preset in src/config/presets/models/\n` +
          `  3. File an issue at https://github.com/clocksmith/doppler/issues\n\n` +
          `Supported model families: gemma2, gemma3, embeddinggemma, llama3, qwen3, mixtral, deepseek, mamba`
        );
      }
      preset = resolvePreset(presetId);
      modelType = preset.modelType;
      if (!modelType) {
        throw new Error(`Preset "${presetId}" missing modelType`);
      }
      const hfConfig = (config || (modelInfo.format === 'gguf' ? null : modelInfo.config));
      const ggufConfig = modelInfo.format === 'gguf' ? modelInfo.config : undefined;
      const architecture = extractArchitecture(hfConfig || {}, ggufConfig);
      headDim = architecture.headDim;
      if (!headDim) {
        throw new Error('Missing headDim in architecture');
      }
      tensorNames = modelInfo.tensors.map((tensor) => tensor.name);
    }
    const tensors = modelInfo.tensors;
    const totalTensorBytes = tensors.reduce((sum, tensor) => sum + (tensor.size || 0), 0);
    const weightOverride = resolvedConverterConfig.quantization?.weights ?? null;
    const sourceQuantization = weightOverride || modelInfo.quantization || inferQuantizationFromTensors(tensors);
    if (!sourceQuantization) {
      throw new Error('Missing quantization for model conversion');
    }
    const embedDtypeRaw = findTensorDtype(tensors, isEmbeddingTensorName);
    const lmHeadDtypeRaw = findTensorDtype(tensors, isLmHeadTensorName);
    const visionPatterns = ['vision_', 'vision_tower', 'vision_model', 'image_encoder'];
    const audioPatterns = ['audio_', 'audio_encoder', 'whisper', 'wav2vec'];
    const projectorPatterns = ['multi_modal_projector', 'mm_projector', 'projector'];
    const hasVision = tensors.some((t) => visionPatterns.some((pattern) => t.name.toLowerCase().includes(pattern)));
    const hasAudio = tensors.some((t) => audioPatterns.some((pattern) => t.name.toLowerCase().includes(pattern)));
    const hasProjector = tensors.some((t) => projectorPatterns.some((pattern) => t.name.toLowerCase().includes(pattern)));
    const quantizationInfo = buildQuantizationInfo(
      resolvedConverterConfig,
      sourceQuantization,
      embedDtypeRaw,
      lmHeadDtypeRaw,
      hasVision,
      hasAudio,
      hasProjector,
      rawConfig
    );
    const manifestQuantization = resolveManifestQuantization(
      resolvedConverterConfig.quantization.weights ?? null,
      sourceQuantization
    );
    if (!diffusionInfo) {
      const inferredHeadDim = headDim ?? preset?.architecture?.headDim ?? null;
      const names = tensorNames ?? tensors.map((tensor) => tensor.name);
      manifestInference = buildManifestInference(preset, rawConfig, inferredHeadDim, quantizationInfo, names);
    }

    const detectedModelId = extractModelId(files, config);
    const baseModelId = userModelId ?? resolvedConverterConfig.output?.modelId ?? detectedModelId;
    const resolvedModelId = baseModelId
      ? resolveModelId(baseModelId, detectedModelId ?? baseModelId, quantizationInfo.variantTag)
      : null;
    modelId = resolvedModelId ? sanitizeModelId(resolvedModelId) : null;
    if (!modelId) {
      throw new Error('Missing modelId. Provide modelId explicitly or include a name in config files.');
    }

    tuneConverterConfig(resolvedConverterConfig, totalTensorBytes, modelType);

    const chunkSizeBytes = resolvedConverterConfig.streaming.chunkSizeBytes;
    if (!chunkSizeBytes || chunkSizeBytes <= 0) {
      throw new Error('Invalid converter streaming chunk size');
    }

    const collectChunks = async (chunks) => {
      const buffers = [];
      let total = 0;
      for await (const chunk of chunks) {
        const bytes = chunk instanceof Uint8Array ? chunk : new Uint8Array(chunk);
        if (bytes.length === 0) continue;
        buffers.push(bytes);
        total += bytes.length;
      }
      const output = new Uint8Array(total);
      let offset = 0;
      for (const buffer of buffers) {
        output.set(buffer, offset);
        offset += buffer.length;
      }
      return output;
    };

    const sourceChunksFor = (tensor) => {
      if (modelInfo.format === 'gguf' && tensor.source) {
        return streamSourceRange(tensor.source, tensor.offset, tensor.size, chunkSizeBytes);
      }
      return streamTensorData(tensor, chunkSizeBytes);
    };

    const tensorPlans = modelInfo.tensors.map((tensor) => {
      const targetDtype = resolveTensorDtype(tensor.name, tensor.shape, tensor.dtype, quantizationInfo);
      const q4kLayout = targetDtype === 'Q4_K_M'
        ? resolveQ4KLayout(tensor.name, tensor.shape, quantizationInfo)
        : null;
      const numElements = tensor.shape.reduce((a, b) => a * b, 1);
      const targetSize = targetDtype === 'Q4_K_M'
        ? getQ4KOutputSize(tensor.shape, q4kLayout)
        : targetDtype === 'F16'
          ? numElements * 2
          : targetDtype === 'F32'
            ? numElements * 4
            : tensor.size;

      const getQ4KData = async () => {
        const chunks = createQ4KChunkStream(
          sourceChunksFor(tensor),
          tensor.dtype,
          tensor.shape,
          q4kLayout,
          chunkSizeBytes
        );
        return collectChunks(chunks);
      };

      const getF16Data = async () => {
        const chunks = createF16ChunkStream(sourceChunksFor(tensor), tensor.dtype);
        return collectChunks(chunks);
      };

      let getData;
      let getChunks;

      if (targetDtype === 'Q4_K_M') {
        getData = getQ4KData;
        getChunks = () => createQ4KChunkStream(
          sourceChunksFor(tensor),
          tensor.dtype,
          tensor.shape,
          q4kLayout,
          chunkSizeBytes
        );
      } else if (targetDtype === 'F16' && tensor.dtype !== 'F16') {
        getData = getF16Data;
        getChunks = () => createF16ChunkStream(sourceChunksFor(tensor), tensor.dtype);
      } else {
        getData = async () => {
          const data = await readTensorData(tensor);
          return new Uint8Array(data);
        };
        getChunks = () => sourceChunksFor(tensor);
      }

      return {
        name: tensor.name,
        shape: tensor.shape,
        dtype: targetDtype,
        size: targetSize,
        offset: tensor.offset,
        getData,
        getChunks,
      };
    });

    const totalSizeBytes = tensorPlans.reduce((sum, tensor) => sum + tensor.size, 0);
    const spaceCheck = await checkSpaceAvailable(totalSizeBytes);
    if (!spaceCheck.hasSpace) {
      throw new QuotaExceededError(totalSizeBytes, spaceCheck.info.available);
    }

    reportProgress({
      stage: ConvertStage.PARSING,
      message: `Model: ${modelId}`,
      modelId,
      tensorCount: modelInfo.tensors.length,
      totalSize: formatBytes(totalSizeBytes),
    });

    await openModelStore(modelId);

    if (diffusionAuxFiles && diffusionAuxFiles.length > 0) {
      for (const asset of diffusionAuxFiles) {
        await saveAuxFile(asset.name, asset.data);
      }
    }

    const hashAlgorithm = resolvedConverterConfig.manifest.hashAlgorithm;

    // Create shard I/O adapter
    const shardIO = {
      writeShard: async (index, data) => {
        const writer = await createConversionShardWriter(index);
        try {
          await writer.write(data);
          await writer.close();
        } catch (error) {
          await writer.abort();
          throw error;
        }
        return computeHash(data, hashAlgorithm);
      },
      computeHash: (data) => computeHash(data, hashAlgorithm),
      createShardWriter: (index) => createConversionShardWriter(index),
      createHasher: () => createStreamingHasher(hashAlgorithm),
    };

    // Create shard packer
    const packer = new ShardPacker(shardIO, {
      modelType,
      shardSize: resolvedConverterConfig.sharding.shardSizeBytes,
      hashAlgorithm,
    });

    // Prepare tensors for packing
    const packerTensors = tensorPlans;

    // Pack tensors into shards
    reportProgress({
      stage: ConvertStage.WRITING,
      message: 'Packing tensors...',
    });

    const packTimer = createStageTimer('Pack shards');
    const packResult = await packer.pack(packerTensors, {
      onProgress: (current, total, tensorName) => {
        reportProgress({
          stage: ConvertStage.WRITING,
          message: `Processing ${tensorName}`,
          current,
          total,
          percent: Math.round((current / total) * 100),
        });
      },
      signal,
    });
    packTimer.stop(
      `${packResult.shards.length} shards, ${formatBytes(packResult.totalSize)}`
    );

    // Convert pack result to expected format
    const result = {
      totalSize: packResult.totalSize,
      tensorLocations: packResult.tensors,
    };

    // Copy shard infos
    for (const shard of packResult.shards) {
      shardInfos.push(shard);
    }

    if (signal?.aborted) throw new DOMException('Cancelled', 'AbortError');

    // Create manifest using shared function
    reportProgress({
      stage: ConvertStage.MANIFEST,
      message: 'Creating manifest...',
    });

    // Convert to ParsedModel format for createManifest
    const parsedModel = {
      tensors: tensorPlans.map(t => ({
        name: t.name,
        shape: t.shape,
        dtype: t.dtype,
        size: t.size,
        offset: t.offset,
      })),
      config: (config || modelInfo.config || {}),
      architecture: modelInfo.architecture,
      quantization: manifestQuantization,
      tokenizerJson,
      tokenizerConfig,
      tokenizerModel: tokenizerModel ? 'tokenizer.model' : null,
    };

    const manifestTimer = createStageTimer('Manifest');
    const manifest = createManifest(
      modelId,
      parsedModel,
      shardInfos,
      result.tensorLocations,
      {
        source: 'browser-converter',
        inference: manifestInference,
        modelType,
        quantization: manifestQuantization,
        quantizationInfo,
        hashAlgorithm,
        architecture: diffusionArchitecture ?? undefined,
        eosTokenId: diffusionEosTokenId,
      }
    );

    manifest.groups = packResult.groups;
    manifest.tensorCount = packResult.tensorCount;
    if (manifest.tokenizer) {
      if (manifest.tokenizer.type === 'bundled' || manifest.tokenizer.type === 'huggingface') {
        manifest.tokenizer.file = manifest.tokenizer.file ?? 'tokenizer.json';
      }
      if (manifest.tokenizer.type === 'sentencepiece') {
        manifest.tokenizer.sentencepieceModel = manifest.tokenizer.sentencepieceModel ?? 'tokenizer.model';
      }
    }

    if (tokenizerJson) {
      await saveTokenizer(JSON.stringify(tokenizerJson));
    }
    if (tokenizerModel) {
      await saveTokenizerModel(tokenizerModel);
    }

    // Save manifest
    await saveManifest(JSON.stringify(manifest, null, 2));

    try {
      await registerModel({
        modelId,
        totalSize: manifest.totalSize ?? result.totalSize,
        quantization: manifest.quantization,
        hashAlgorithm: manifest.hashAlgorithm,
        backend: getStorageBackendType(),
      });
    } catch {
      // Registry is optional; ignore failures
    }
    manifestTimer.stop();

    reportProgress({
      stage: ConvertStage.COMPLETE,
      message: 'Conversion complete!',
      modelId,
      shardCount: shardInfos.length,
      totalSize: formatBytes(result.totalSize),
    });

    if (cleanupTasks.length > 0) {
      await Promise.allSettled(cleanupTasks.map((task) => task()));
    }

    const totalMs = performance.now() - conversionStart;
    log.info(
      'Convert',
      `Complete: ${formatBytes(result.totalSize)} in ${totalMs.toFixed(0)}ms`
    );
    trace.perf('Convert total', {
      ms: totalMs,
      tensors: modelInfo.tensors.length,
      totalSize: result.totalSize,
      shardCount: shardInfos.length,
    });

    return modelId;
  } catch (error) {
    const totalMs = performance.now() - conversionStart;
    log.error('Convert', `Failed after ${totalMs.toFixed(0)}ms: ${error.message}`);
    // Cleanup on error
    if (modelId) {
      try {
        await deleteModel(modelId);
      } catch {
        // Ignore cleanup errors
      }
    }

    reportProgress({
      stage: ConvertStage.ERROR,
      message: error.message,
      error: error,
    });

    throw error;
  }
}


async function parseGGUFModel(
  file,
  onProgress,
  signal
) {
  onProgress?.({
    stage: ConvertStage.PARSING,
    message: 'Parsing GGUF header...',
  });

  const source = normalizeTensorSource(file);
  const ggufInfo = await parseGGUFHeaderFromSource(source);

  return {
    format: 'gguf',
    tensors: ggufInfo.tensors.map((t) => ({
      ...t,
      file: source.file,
      source,
      offset: t.offset,
    })),
    config: ggufInfo.config,
    architecture: ggufInfo.architecture,
    quantization: ggufInfo.quantization,
    tensorDataOffset: ggufInfo.tensorDataOffset,
    file: source.file,
    source,
  };
}


async function* streamSourceRange(source, offset, size, chunkSize) {
  let cursor = offset;
  const end = offset + size;

  while (cursor < end) {
    const next = Math.min(cursor + chunkSize, end);
    const buffer = await source.readRange(cursor, next - cursor);
    yield new Uint8Array(buffer);
    cursor = next;
  }
}


function extractModelId(files, config) {
  // Try config first
  if (config?._name_or_path) {
    const parts = config._name_or_path.split('/');
    return parts[parts.length - 1];
  }

  const safetensorsFiles = files.filter((f) => getBaseName(getFilePath(f)).toLowerCase().endsWith('.safetensors'));
  const rootSafetensors = safetensorsFiles.find((f) => !normalizePath(getFilePath(f)).includes('/'));
  const stFile = rootSafetensors || safetensorsFiles.find((f) => {
    const base = getBaseName(getFilePath(f)).toLowerCase();
    return !base.startsWith('model-') && !base.includes('-of-');
  }) || safetensorsFiles[0];
  if (stFile) {
    const base = getBaseName(getFilePath(stFile));
    return base.replace(/\.safetensors$/, '').replace(/model[-_.]?/, '');
  }

  // Try GGUF file name
  const ggufFile = files.find((f) => f.name.endsWith('.gguf'));
  if (ggufFile) {
    return ggufFile.name.replace(/\.gguf$/, '');
  }

  return null;
}

// ============================================================================
// File Picker Utilities
// ============================================================================


export async function pickModelFiles() {
  // Try directory picker first (for HuggingFace models)
  if ('showDirectoryPicker' in window) {
    try {
      const dirHandle = await window.showDirectoryPicker({
        mode: 'read',
      });
      return await collectFilesFromDirectory(dirHandle);
    } catch (e) {
      if (e.name === 'AbortError') throw e;
      // Fall back to file picker
    }
  }

  // Fall back to file picker
  if ('showOpenFilePicker' in window) {
    const handles = await window.showOpenFilePicker({
      multiple: true,
      types: [
        {
          description: 'Model files',
          accept: {
            'application/octet-stream': ['.gguf', '.safetensors', '.bin'],
            'application/json': ['.json'],
          },
        },
      ],
    });
    return Promise.all(handles.map((h) => h.getFile()));
  }

  // Ultimate fallback: input element
  return new Promise((resolve) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.multiple = true;
    input.accept = '.gguf,.safetensors,.json,.bin';
    input.onchange = () => {
      resolve(Array.from(input.files || []));
    };
    input.click();
  });
}


async function collectFilesFromDirectory(
  dirHandle,
  files = []
) {
  const entries = dirHandle.values();
  for await (const entry of entries) {
    if (entry.kind === 'file') {
      const file = await entry.getFile();
      // Only include relevant files
      if (
        file.name.endsWith('.safetensors') ||
        file.name.endsWith('.gguf') ||
        file.name.endsWith('.json') ||
        file.name === 'tokenizer.model'
      ) {
        files.push(file);
      }
    }
  }
  return files;
}

export default convertModel;
