

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
import { classifyTensorRole } from '../storage/rdrr-format.js';
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
import {
  resolveTensorDtype,
  resolveQ4KLayout,
  getQ4KOutputSize,
  createQ4KChunkStream,
  createF16ChunkStream,
  resolveEffectiveQuantizationInfo,
  resolveManifestQuantization,
} from './quantization.js';

// Import shared types and functions from convert-core
import {
  ConvertStage,
  formatBytes,
  extractArchitecture,
  createManifest,
} from '../converter/core.js';
import {
  inferSourceWeightQuantization,
  resolveConversionPlan,
  resolveConvertedModelId,
} from '../converter/conversion-plan.js';
import { parseDiffusionModel as parseSharedDiffusionModel } from '../converter/parsers/diffusion.js';
import { parseGGUFModel as parseSharedGGUFModel } from '../converter/parsers/gguf.js';

import { createConverterConfig } from '../config/index.js';
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
  return parseSharedDiffusionModel({
    onProgress,
    signal,
    findExistingSuffix(suffixes) {
      const file = pickFirstBySuffix(files, suffixes);
      return file ? normalizePath(getFilePath(file)) : null;
    },
    async readJson(suffix, label = 'json') {
      const file = findFileBySuffix(files, suffix);
      if (!file) {
        throw new Error(`Missing ${label} (${suffix})`);
      }
      return parseJsonFile(file, label);
    },
    async readText(suffix, label = 'text') {
      const file = findFileBySuffix(files, suffix);
      if (!file) {
        throw new Error(`Missing ${label} (${suffix})`);
      }
      return readTextFile(file, label);
    },
    async readBinary(suffix, label = 'binary') {
      const file = findFileBySuffix(files, suffix);
      if (!file || typeof file.arrayBuffer !== 'function') {
        throw new Error(`Missing ${label} (${suffix})`);
      }
      return file.arrayBuffer();
    },
    async parseSingleSafetensors(suffix) {
      const file = findFileBySuffix(files, suffix);
      if (!file) {
        throw new Error(`Missing safetensors file (${suffix})`);
      }
      return parseSafetensorsFile(file);
    },
    async parseShardedSafetensors(indexSuffix, indexJson, componentId) {
      const weightMap = indexJson?.weight_map || {};
      const shardNames = Array.from(new Set(Object.values(weightMap)));
      if (shardNames.length === 0) {
        throw new Error(`No shards listed in ${componentId} index file`);
      }
      const baseDir = indexSuffix.includes('/') ? indexSuffix.split('/').slice(0, -1).join('/') : '';
      const shardFiles = shardNames.map((name) => {
        const suffix = baseDir ? `${baseDir}/${name}` : name;
        return findFileBySuffix(files, suffix);
      }).filter(Boolean);
      if (shardFiles.length !== shardNames.length) {
        throw new Error(`Missing shard files for ${componentId} (${shardFiles.length}/${shardNames.length} found)`);
      }
      return parseSafetensorsSharded(shardFiles, indexJson);
    },
  });
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

    const rawConfig = (config || modelInfo.config || {});
    const tensors = modelInfo.tensors;
    const totalTensorBytes = tensors.reduce((sum, tensor) => sum + (tensor.size || 0), 0);
    let architectureConfig = null;
    if (!diffusionInfo) {
      const hfConfig = (config || (modelInfo.format === 'gguf' ? null : modelInfo.config));
      const ggufConfig = modelInfo.format === 'gguf' ? modelInfo.config : undefined;
      architectureConfig = extractArchitecture(hfConfig || {}, ggufConfig);
    }
    const weightOverride = resolvedConverterConfig.quantization?.weights ?? null;
    const sourceQuantization = modelInfo.quantization || weightOverride || inferSourceWeightQuantization(tensors);
    const plan = resolveConversionPlan({
      rawConfig,
      tensors,
      converterConfig: resolvedConverterConfig,
      sourceQuantization,
      modelKind: diffusionInfo ? 'diffusion' : 'transformer',
      architectureHint: modelInfo.architecture,
      architectureConfig,
    });
    const modelType = plan.modelType;
    const quantizationInfo = plan.quantizationInfo;
    const manifestQuantization = plan.manifestQuantization;
    const manifestInference = plan.manifestInference;

    const detectedModelId = extractModelId(files, config);
    modelId = resolveConvertedModelId({
      explicitModelId: userModelId,
      converterConfig: resolvedConverterConfig,
      detectedModelId,
      quantizationInfo,
    });
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
        role: classifyTensorRole(tensor.name),
        layout: q4kLayout,
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
    const effectiveQuantizationInfo = resolveEffectiveQuantizationInfo(
      quantizationInfo,
      tensorPlans
    );
    const effectiveManifestQuantization = resolveManifestQuantization(
      effectiveQuantizationInfo.weights,
      manifestQuantization
    );

    const manifest = createManifest(
      modelId,
      parsedModel,
      shardInfos,
      result.tensorLocations,
      {
        source: 'browser-converter',
        inference: manifestInference,
        modelType,
        quantization: effectiveManifestQuantization,
        quantizationInfo: effectiveQuantizationInfo,
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
  return parseSharedGGUFModel({
    file,
    parseGGUFHeaderFromSource,
    normalizeTensorSource,
    onProgress,
    signal,
  });
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
