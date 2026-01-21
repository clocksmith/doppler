

import { parseGGUFHeader } from './gguf-parser-browser.js';
import { canStreamFile } from './file-picker.js';
import {
  initStorage,
  openModelStore,
  saveManifest,
  deleteModel,
  computeHash,
  getStorageBackendType,
} from '../storage/shard-manager.js';
import {
  RDRR_VERSION,
  generateShardFilename,
} from '../storage/rdrr-format.js';
import { log } from '../debug/index.js';
import { createConverterConfig, detectPreset, resolvePreset } from '../config/index.js';
import { extractArchitecture } from '../converter/core.js';
import { buildManifestInference } from '../converter/manifest-inference.js';
import { HEADER_READ_SIZE } from '../config/schema/index.js';
import { registerModel } from '../storage/registry.js';

// ============================================================================
// Types
// ============================================================================


export const ImportStage = {
  PARSING: 'parsing',
  SHARDING: 'sharding',
  WRITING: 'writing',
  COMPLETE: 'complete',
  ERROR: 'error',
};

// ============================================================================
// Helper Functions
// ============================================================================


function sanitizeModelId(name) {
  const sanitized = name
    .toLowerCase()
    .replace(/[^a-z0-9_-]/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 64);
  return sanitized || null;
}

// ============================================================================
// Main Import Function
// ============================================================================


export async function importGGUFFile(
  file,
  { onProgress, signal, modelId: userModelId, converterConfig } = {}
) {
  let modelId = null;
  let modelDir = null;
  const shardInfos = [];
  const resolvedConverterConfig = converterConfig || createConverterConfig();
  const shardingConfig = resolvedConverterConfig.sharding;

  try {
    // Initialize OPFS
    await initStorage();

    // Report parsing stage
    onProgress?.({
      stage: ImportStage.PARSING,
      message: 'Parsing GGUF header...',
      filename: file.name,
    });

    // Check for abort
    if (signal?.aborted) {
      throw new DOMException('Import cancelled', 'AbortError');
    }

    // Read header portion for parsing
    const headerBlob = file.slice(0, Math.min(HEADER_READ_SIZE, file.size));
    const headerBuffer = await headerBlob.arrayBuffer();
    const ggufInfo = parseGGUFHeader(headerBuffer);

    // Generate model ID from filename or GGUF metadata
    const ggufName = ggufInfo.modelName !== 'unknown'
      ? ggufInfo.modelName
      : file.name.replace(/\.gguf$/i, '');
    modelId = sanitizeModelId(userModelId || ggufName);
    if (!modelId) {
      throw new Error('Missing modelId. Provide modelId explicitly or include a name in GGUF metadata.');
    }

    onProgress?.({
      stage: ImportStage.PARSING,
      message: `Model: ${modelId}`,
      modelId,
      architecture: ggufInfo.architecture,
      quantization: ggufInfo.quantization,
    });

    // Open model directory in OPFS
    modelDir = await openModelStore(modelId);
    if (!modelDir) {
      throw new Error('OPFS required for GGUF import');
    }

    // Calculate expected shard count
    const totalDataSize = file.size - ggufInfo.tensorDataOffset;
    const shardSizeBytes = shardingConfig?.shardSizeBytes;
    if (!shardSizeBytes || shardSizeBytes <= 0) {
      throw new Error('Invalid converter shard size configuration');
    }
    const expectedShards = Math.ceil(totalDataSize / shardSizeBytes);

    onProgress?.({
      stage: ImportStage.SHARDING,
      message: `Preparing ${expectedShards} shards...`,
      current: 0,
      total: expectedShards,
      percent: 0,
    });

    // Check for abort
    if (signal?.aborted) {
      throw new DOMException('Import cancelled', 'AbortError');
    }

    // Stream the file and create shards
    if (canStreamFile(file)) {
      await streamToShards(file, ggufInfo, modelDir, shardInfos, shardSizeBytes, {
        onProgress,
        signal,
      });
    } else {
      // Fallback for browsers without streaming
      await bufferToShards(file, ggufInfo, modelDir, shardInfos, shardSizeBytes, {
        onProgress,
        signal,
      });
    }

    // Check for abort before finalizing
    if (signal?.aborted) {
      throw new DOMException('Import cancelled', 'AbortError');
    }

    // Create manifest
    const manifest = createManifest(
      ggufInfo,
      shardInfos,
      modelId,
      resolvedConverterConfig,
      shardSizeBytes
    );

    onProgress?.({
      stage: ImportStage.WRITING,
      message: 'Saving manifest...',
    });

    // Save manifest to OPFS
    await saveManifest(JSON.stringify(manifest, null, 2));

    try {
      await registerModel({
        modelId,
        totalSize: file.size,
        quantization: ggufInfo.quantization,
        hashAlgorithm: converterConfig.manifest.hashAlgorithm,
        backend: getStorageBackendType(),
      });
    } catch {
      // Registry is optional; ignore failures
    }

    onProgress?.({
      stage: ImportStage.COMPLETE,
      message: 'Import complete!',
      modelId,
      shardCount: shardInfos.length,
      totalSize: file.size,
    });

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
      stage: ImportStage.ERROR,
      message: error.message,
      error: error,
    });

    throw error;
  }
}


async function streamToShards(
  file,
  ggufInfo,
  modelDir,
  shardInfos,
  shardSizeBytes,
  { onProgress, signal }
) {
  const tensorDataOffset = ggufInfo.tensorDataOffset;
  const totalDataSize = file.size - tensorDataOffset;
  const expectedShards = Math.ceil(totalDataSize / shardSizeBytes);

  // Slice to just tensor data
  const tensorBlob = file.slice(tensorDataOffset);
  const stream = tensorBlob.stream();
  const reader = stream.getReader();

  let shardIndex = 0;
  let shardBuffer = new Uint8Array(shardSizeBytes);
  let shardOffset = 0;
  let totalProcessed = 0;

  try {
    while (true) {
      // Check for abort
      if (signal?.aborted) {
        reader.cancel();
        throw new DOMException('Import cancelled', 'AbortError');
      }

      const { done, value } = await reader.read();

      if (done) {
        // Write final partial shard if any data remains
        if (shardOffset > 0) {
          await writeShard(
            modelDir,
            shardIndex,
            shardBuffer.slice(0, shardOffset),
            shardInfos,
            shardSizeBytes,
            converterConfig.manifest.hashAlgorithm
          );
          shardIndex++;
        }
        break;
      }

      // Process chunk
      let chunkOffset = 0;
      while (chunkOffset < value.length) {
        const remaining = shardSizeBytes - shardOffset;
        const toCopy = Math.min(remaining, value.length - chunkOffset);

        shardBuffer.set(value.subarray(chunkOffset, chunkOffset + toCopy), shardOffset);
        shardOffset += toCopy;
        chunkOffset += toCopy;
        totalProcessed += toCopy;

        // Shard full, write it
        if (shardOffset === shardSizeBytes) {
          await writeShard(
            modelDir,
            shardIndex,
            shardBuffer,
            shardInfos,
            shardSizeBytes,
            converterConfig.manifest.hashAlgorithm
          );

          shardIndex++;
          shardBuffer = new Uint8Array(shardSizeBytes);
          shardOffset = 0;

          onProgress?.({
            stage: ImportStage.SHARDING,
            message: `Writing shard ${shardIndex}/${expectedShards}`,
            current: shardIndex,
            total: expectedShards,
            percent: Math.round((totalProcessed / totalDataSize) * 100),
          });
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}


async function bufferToShards(
  file,
  ggufInfo,
  modelDir,
  shardInfos,
  shardSizeBytes,
  { onProgress, signal }
) {
  const tensorDataOffset = ggufInfo.tensorDataOffset;
  const totalDataSize = file.size - tensorDataOffset;
  const expectedShards = Math.ceil(totalDataSize / shardSizeBytes);

  log.warn('GGUF Import', 'Using buffer fallback - large files may cause memory issues');

  let shardIndex = 0;
  let offset = tensorDataOffset;

  while (offset < file.size) {
    // Check for abort
    if (signal?.aborted) {
      throw new DOMException('Import cancelled', 'AbortError');
    }

    const end = Math.min(offset + shardSizeBytes, file.size);
    const blob = file.slice(offset, end);
    const buffer = await blob.arrayBuffer();
    const data = new Uint8Array(buffer);

      await writeShard(modelDir, shardIndex, data, shardInfos, shardSizeBytes, converterConfig.manifest.hashAlgorithm);

    shardIndex++;
    offset = end;

    onProgress?.({
      stage: ImportStage.SHARDING,
      message: `Writing shard ${shardIndex}/${expectedShards}`,
      current: shardIndex,
      total: expectedShards,
      percent: Math.round(((offset - tensorDataOffset) / totalDataSize) * 100),
    });
  }
}


async function writeShard(
  modelDir,
  shardIndex,
  data,
  shardInfos,
  shardSizeBytes,
  hashAlgorithm
) {
  const filename = generateShardFilename(shardIndex);
  const hash = await computeHash(data, hashAlgorithm);

  // Get file handle and write
  const fileHandle = await modelDir.getFileHandle(filename, { create: true });
  const writable = await fileHandle.createWritable();
  await writable.write(new Uint8Array(data.buffer, data.byteOffset, data.length));
  await writable.close();

  // Record shard info
  shardInfos.push({
    index: shardIndex,
    filename,
    size: data.length,
    hash: hash, // SHA-256 hash
    offset: shardIndex * shardSizeBytes,
  });
}


function createManifest(
  ggufInfo,
  shardInfos,
  modelId,
  converterConfig,
  shardSizeBytes
) {
  const config = ggufInfo.config;
  if (!ggufInfo.architecture) {
    throw new Error('Missing GGUF architecture');
  }
  if (!ggufInfo.quantization) {
    throw new Error('Missing GGUF quantization');
  }

  const architecture = extractArchitecture({}, config);

  // Build MoE config if applicable
  let moeConfig = null;
  const expertCount = config.expertCount || config.num_local_experts || config.num_experts;
  if (expertCount) {
    const expertsPerToken = (
      config.expertUsedCount ||
      config.num_experts_per_tok ||
      config.num_experts_per_token ||
      config.experts_per_token
    );
    if (!expertsPerToken) {
      throw new Error('Missing expertsPerToken in GGUF config');
    }
    const expertFormat = config.expertFormat || config.expert_format;
    if (!expertFormat) {
      throw new Error('Missing expertFormat in GGUF config');
    }
    moeConfig = {
      numExperts: expertCount,
      numExpertsPerToken: expertsPerToken,
      expertFormat: expertFormat,
    };
  }

  // Calculate total size from shards
  const totalSize = shardInfos.reduce((sum, s) => sum + s.size, 0);

  const rawConfig = {
    model_type: ggufInfo.architecture,
    architectures: [ggufInfo.architecture],
  };
  if (config.ropeFreqBase) {
    rawConfig.rope_theta = config.ropeFreqBase;
  }
  if (config.ropeScalingType || config.ropeScalingFactor) {
    rawConfig.rope_scaling = {
      type: config.ropeScalingType ?? undefined,
      factor: config.ropeScalingFactor ?? undefined,
    };
  }

  const presetOverride = converterConfig.presets?.model;
  const presetId = presetOverride || detectPreset(rawConfig, ggufInfo.architecture);
  if (presetId === 'transformer') {
    const modelType = rawConfig.model_type ?? 'unknown';
    throw new Error(
      `Unknown model family: architecture="${ggufInfo.architecture || 'unknown'}", model_type="${modelType}"\n\n` +
      `DOPPLER requires a known model preset to generate correct inference config.\n` +
      `The manifest-first architecture does not support generic defaults.\n\n` +
      `Options:\n` +
      `  1. Wait for official support of this model family\n` +
      `  2. Create a custom preset in src/config/presets/models/\n` +
      `  3. File an issue at https://github.com/clocksmith/doppler/issues\n\n` +
      `Supported model families: gemma2, gemma3, llama3, qwen3, mixtral, deepseek, mamba, gpt-oss`
    );
  }
  const preset = resolvePreset(presetId);
  const headDim = architecture.headDim;
  if (!headDim) {
    throw new Error('Missing headDim in GGUF architecture');
  }
  const quantizationInfo = ggufInfo.quantization
    ? { weights: ggufInfo.quantization, compute: converterConfig.quantization.computePrecision }
    : null;
  const inference = buildManifestInference(preset, rawConfig, headDim, quantizationInfo);

  // Build tensor location map
  // Maps each tensor to its shard(s) and offset within shard
  const tensors = buildTensorLocations(ggufInfo.tensors, ggufInfo.tensorDataOffset, shardSizeBytes);

  return {
    version: RDRR_VERSION,
    modelId,
    modelType: ggufInfo.architecture,
    quantization: ggufInfo.quantization,
    architecture,
    moeConfig,
    shards: shardInfos,
    tensors,
    totalSize,
    hashAlgorithm: converterConfig.manifest.hashAlgorithm,
    inference,
    metadata: {
      source: 'browser-import',
      originalFile: ggufInfo.modelName,
      importedAt: new Date().toISOString(),
      ggufVersion: ggufInfo.version,
    },
  };
}


function buildTensorLocations(
  ggufTensors,
  tensorDataOffset,
  shardSizeBytes
) {
  const tensors = {};

  for (const tensor of ggufTensors) {
    // Position relative to tensor data start (not file start)
    const relativeOffset = tensor.offset - tensorDataOffset;

    // Which shard does this tensor start in?
    const startShard = Math.floor(relativeOffset / shardSizeBytes);
    const offsetInShard = relativeOffset % shardSizeBytes;

    // Does tensor fit entirely in one shard?
    const endOffset = offsetInShard + tensor.size;

    if (endOffset <= shardSizeBytes) {
      // Tensor fits in single shard
      tensors[tensor.name] = {
        shard: startShard,
        offset: offsetInShard,
        size: tensor.size,
        shape: tensor.shape,
        dtype: tensor.dtype,
      };
    } else {
      // Tensor spans multiple shards - create spans array
      const spans = [];
      let remaining = tensor.size;
      let currentShard = startShard;
      let currentOffset = offsetInShard;

      while (remaining > 0) {
        const availableInShard = shardSizeBytes - currentOffset;
        const chunkSize = Math.min(remaining, availableInShard);

        spans.push({
          shardIndex: currentShard,
          offset: currentOffset,
          size: chunkSize,
        });

        remaining -= chunkSize;
        currentShard++;
        currentOffset = 0; // Next shard starts at offset 0
      }

      tensors[tensor.name] = {
        spans,
        size: tensor.size,
        shape: tensor.shape,
        dtype: tensor.dtype,
      };
    }
  }

  return tensors;
}

export default importGGUFFile;
