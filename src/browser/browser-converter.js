

import { parseGGUFHeaderFromSource } from './gguf-parser-browser.js';
import { isTensorSource, normalizeTensorSource } from './tensor-source-file.js';
import { createRemoteTensorSource } from './tensor-source-download.js';
import {
  parseSafetensorsFile,
  parseSafetensorsSharded,
  parseConfigJson,
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
  deleteModel,
  createConversionShardWriter,
  computeHash,
  createStreamingHasher,
  getStorageBackendType,
} from '../storage/shard-manager.js';
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

// Re-export types for consumers
export {
  ConvertStage,
};

export function isConversionSupported() {
  return isOPFSAvailable() || isIndexedDBAvailable();
}

export async function createRemoteModelSources(urls, options = {}) {
  if (!Array.isArray(urls) || urls.length === 0) {
    throw new Error('Remote conversion requires at least one URL.');
  }

  const sources = [];
  for (const url of urls) {
    if (typeof url !== 'string' || url.length === 0) {
      throw new Error('Remote conversion URLs must be non-empty strings.');
    }
    const result = await createRemoteTensorSource(url, options);
    sources.push(result.source);
  }

  return sources;
}

// ============================================================================
// Main Convert Function
// ============================================================================


function inferQuantizationFromTensors(tensors) {
  const weightDtypes = new Set();
  for (const tensor of tensors) {
    if (!tensor?.name || typeof tensor.dtype !== 'string') continue;
    if (!tensor.name.includes('.weight')) continue;
    weightDtypes.add(tensor.dtype.toUpperCase());
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

export async function convertModel(files, options = {}) {
  const { modelId: userModelId, onProgress, signal, converterConfig } = options;
  const resolvedConverterConfig = converterConfig || createConverterConfig();

  let modelId = null;
  const shardInfos = [];
  const cleanupTasks = [];

  try {
    if (!isOPFSAvailable() && !isIndexedDBAvailable()) {
      throw new Error('No supported storage backend available for browser conversion. Supported: opfs, indexeddb.');
    }

    // Initialize storage
    await initStorage();
    const persistence = await requestPersistence();
    const backendType = getStorageBackendType();
    onProgress?.({
      stage: ConvertStage.DETECTING,
      message: `Storage backend: ${backendType ?? 'unknown'}`,
      backend: backendType,
      persistence,
    });

    // Detect format
    onProgress?.({
      stage: ConvertStage.DETECTING,
      message: 'Detecting model format...',
    });

    if (signal?.aborted) throw new DOMException('Cancelled', 'AbortError');

    const format = detectModelFormat(files);
    const auxiliary = getAuxiliaryFiles(files);
    for (const file of files) {
      if (isTensorSource(file) && typeof file.cleanup === 'function') {
        cleanupTasks.push(file.cleanup);
      }
    }
    if (!auxiliary.tokenizer) {
      if (auxiliary.tokenizerModel) {
        throw new Error('tokenizer.model is not supported in browser conversion. Provide tokenizer.json instead.');
      }
      throw new Error('Missing tokenizer.json for browser conversion.');
    }

    onProgress?.({
      stage: ConvertStage.DETECTING,
      message: `Format: ${format.type}`,
      format: format.type,
    });

    // Parse based on format
    let modelInfo;
    let config = null;
    let tokenizerJson = null;

    if (format.type === 'gguf') {
      modelInfo = await parseGGUFModel(format.ggufFile, onProgress, signal);
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

    // Parse tokenizer if available
    if (auxiliary.tokenizer) {
      tokenizerJson = await parseTokenizerJson(auxiliary.tokenizer);
      modelInfo.tokenizerJson = tokenizerJson;
    }

    if (signal?.aborted) throw new DOMException('Cancelled', 'AbortError');

    // Detect model type using preset system
    const rawConfig = (config || modelInfo.config || {});
    const presetId = detectPreset(rawConfig, modelInfo.architecture);
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
        `Supported model families: gemma2, gemma3, llama3, qwen3, mixtral, deepseek, mamba`
      );
    }
    const preset = resolvePreset(presetId);
    const modelType = preset.modelType;
    if (!modelType) {
      throw new Error(`Preset "${presetId}" missing modelType`);
    }
    const hfConfig = (config || (modelInfo.format === 'gguf' ? null : modelInfo.config));
    const ggufConfig = modelInfo.format === 'gguf' ? modelInfo.config : undefined;
    const architecture = extractArchitecture(hfConfig || {}, ggufConfig);
    const headDim = architecture.headDim;
    if (!headDim) {
      throw new Error('Missing headDim in architecture');
    }
    const tensors = modelInfo.tensors;
    const sourceQuantization = modelInfo.quantization || inferQuantizationFromTensors(tensors);
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
    const manifestInference = buildManifestInference(preset, rawConfig, headDim, quantizationInfo);

    const detectedModelId = extractModelId(files, config);
    const baseModelId = userModelId ?? resolvedConverterConfig.output?.modelId ?? detectedModelId;
    const resolvedModelId = baseModelId
      ? resolveModelId(baseModelId, detectedModelId ?? baseModelId, quantizationInfo.variantTag)
      : null;
    modelId = resolvedModelId ? sanitizeModelId(resolvedModelId) : null;
    if (!modelId) {
      throw new Error('Missing modelId. Provide modelId explicitly or include a name in config files.');
    }

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
      if (q4kLayout === 'col') {
        throw new Error(
          'Column-wise Q4_K_M quantization requires full tensor reads and is not supported in browser conversion. ' +
          'Use row layout or provide pre-quantized weights.'
        );
      }
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

    onProgress?.({
      stage: ConvertStage.PARSING,
      message: `Model: ${modelId}`,
      modelId,
      tensorCount: modelInfo.tensors.length,
      totalSize: formatBytes(totalSizeBytes),
    });

    await openModelStore(modelId);

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
    onProgress?.({
      stage: ConvertStage.WRITING,
      message: 'Packing tensors...',
    });

    const packResult = await packer.pack(packerTensors, {
      onProgress: (current, total, tensorName) => {
        onProgress?.({
          stage: ConvertStage.WRITING,
          message: `Processing ${tensorName}`,
          current,
          total,
          percent: Math.round((current / total) * 100),
        });
      },
      signal,
    });

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
    onProgress?.({
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
    };

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
      }
    );

    manifest.groups = packResult.groups;
    manifest.tensorCount = packResult.tensorCount;
    if (manifest.tokenizer) {
      manifest.tokenizer.file = 'tokenizer.json';
    }

    if (tokenizerJson) {
      await saveTokenizer(JSON.stringify(tokenizerJson));
    }

    // Save manifest
    await saveManifest(JSON.stringify(manifest, null, 2));

    onProgress?.({
      stage: ConvertStage.COMPLETE,
      message: 'Conversion complete!',
      modelId,
      shardCount: shardInfos.length,
      totalSize: formatBytes(result.totalSize),
    });

    if (cleanupTasks.length > 0) {
      await Promise.allSettled(cleanupTasks.map((task) => task()));
    }

    return modelId;
  } catch (error) {
    // Cleanup on error
    if (modelId) {
      try {
        await deleteModel(modelId);
      } catch {
        // Ignore cleanup errors
      }
    }

    onProgress?.({
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

  // Try first safetensors file name
  const stFile = files.find((f) => f.name.endsWith('.safetensors'));
  if (stFile) {
    return stFile.name.replace(/\.safetensors$/, '').replace(/model[-_.]?/, '');
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
