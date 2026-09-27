import { getMemoryCapabilities } from '../memory/capability.js';
import { detectUnifiedMemory } from '../memory/unified-detect.js';
import { getHeapManager } from '../memory/heap-manager.js';
import {
  initStorage,
  loadAuxFile,
  computeHash,
} from '../storage/shard-manager.js';
import { clearManifest, setManifest as setCurrentManifest } from '../formats/rdrr/index.js';
import { initDevice, getDevice, getKernelCapabilities } from '../gpu/device.js';
import {
  PersistentBufferSet,
  acquireBuffer,
  isBufferActive,
  releaseBuffer,
  forceBufferPoolReclaim,
} from '../memory/buffer-pool.js';
import { getExpertCache } from './experts/expert-cache.js';
import { log, trace as debugTrace } from '../debug/index.js';
import { isGpuBufferInstance, isWeightBuffer } from '../gpu/weight-buffer.js';
import { createShardCache } from './shard-cache.js';
import { getRuntimeConfig } from '../config/runtime.js';
import { buildTensorLocations } from './shard-resolver.js';
import {
  needsNormWeightOffset,
  resolveWeightLayout,
  requiresCpuF16ToF32MatmulMaterialization,
  shouldStreamLargeWeight,
} from './manifest-config.js';
import { MemoryMonitor } from './memory-monitor.js';
import {
  loadTensorToGPU,
  loadTensorToCPU,
  isLiteRTAffineInt4FusedEligible,
} from './tensors/tensor-loader.js';
import { annotateTensorLoadError } from './tensor-load-error.js';
import { loadEmbeddings } from './embedding-loader.js';
import { loadPerLayerInputWeights } from './per-layer-input-loader.js';
import { loadLayer } from './layer-loader.js';
import { loadFinalWeights } from './final-weights-loader.js';
import { cloneJsonValue } from '../formats/clone-json.js';
import { assertFunctionalDescriptorManifest } from '../formats/rdrr/functional-descriptor.js';
import {
  loadExpert as loadExpertFromModule,
  prefetchExperts as prefetchExpertsFromModule,
  predictNextLayerExperts as predictNextLayerExpertsFromModule,
} from './experts/expert-loader.js';
import { assembleShardData } from './tensors/tensor-reader.js';
import { hasSourceTransform } from './tensors/source-transform.js';

export { load } from './model-load.js';

export async function _loadTensor(name, toGPU = true, silent = false) {
    const location = this.tensorLocations.get(name);
    if (!location) {
      if (!silent) {
        log.warn('Loader', `Tensor not found: ${name}`);
      }
      return null;
    }

    if (name.includes('attn_k') || name.includes('k_proj')) {
      debugTrace.loader(`Loading ${name}: shape=${JSON.stringify(location.shape)}, size=${location.size}, dtype=${location.dtype}, spans=${!!location.spans}`);
    }

    const streamedUpload = toGPU && this._shouldStreamUploadToGPU(location);
    let shardData;
    try {
      const preserveRawSourceBytes = toGPU && isLiteRTAffineInt4FusedEligible(location, {
        gpuCapabilities: this.gpuCapabilities,
      });
      shardData = this._isFunctionalDescriptorLocation(location)
        ? await this._assembleFunctionalDescriptorData(location, name)
        : streamedUpload
        ? await this._assembleShardDataToGpuBuffer(location, name)
        : await this._assembleShardData(location, name, {
            materializeSourceTransform: !preserveRawSourceBytes,
          });
    } catch (error) {
      throw annotateTensorLoadError(error, name, location, {
        tensorLoadStage: streamedUpload ? 'streamShardToGpuBuffer' : 'assembleShardData',
        toGPU,
        streamedUpload,
      });
    }

    if (toGPU) {
      const device = getDevice();
      if (!device) {
        log.warn('Loader', 'GPU device not available; falling back to CPU');
        if (isGpuBufferInstance(shardData)) {
          releaseBuffer(shardData);
          shardData = await this._assembleShardData(location, name);
        }
        return loadTensorToCPU(shardData, location, name);
      }

      
      const allowF32UpcastNonMatmul = this._loadingConfig?.allowF32UpcastNonMatmul;
      if (allowF32UpcastNonMatmul == null) {
        throw new Error('runtime.loading.allowF32UpcastNonMatmul is required.');
      }
      const config = {
        useFusedQ4K: this.useFusedQ4K,
        q4kMaterializationMode: this.q4kMaterializationMode,
        q4kFusedRoles: this.q4kFusedRoles,
        keepF32Weights: this.keepF32Weights,
        keepBF16Weights: this.keepBF16Weights,
        q4kLayout: this.q4kLayout,
        loaderDebug: this._loaderDebug,
        gpuCapabilities: this.gpuCapabilities,
        allowF32UpcastNonMatmul,
      };

      let result;
      try {
        result = await loadTensorToGPU(shardData, location, name, config);
      } catch (error) {
        if (isGpuBufferInstance(shardData)) {
          releaseBuffer(shardData);
        }
        throw annotateTensorLoadError(error, name, location, {
          tensorLoadStage: 'materializeTensorToGPU',
          toGPU: true,
          streamedUpload,
        });
      }

      for (const buffer of result.allocatedBuffers) {
        this.gpuBuffers.add(buffer);
      }

      return result.data;
    }

    if (isGpuBufferInstance(shardData)) {
      // Shouldn't happen (streaming is only used for toGPU), but keep this leak-proof.
      releaseBuffer(shardData);
      shardData = await this._assembleShardData(location, name);
    }
    return loadTensorToCPU(shardData, location, name);
  }

export function _getDescriptorShardFiles(location, name) {
    if (!location?.descriptorManifest) {
      throw new Error(
        `[DopplerLoader] FUNCTIONAL_DESCRIPTOR tensor "${name}" is missing descriptorManifest.`
      );
    }
    const manifest = assertFunctionalDescriptorManifest(
      location.descriptorManifest,
      `FUNCTIONAL_DESCRIPTOR tensor "${name}" descriptorManifest`
    );
    const components = manifest.components;
    if (!components || typeof components !== 'object') {
      throw new Error(
        `[DopplerLoader] FUNCTIONAL_DESCRIPTOR tensor "${name}" descriptorManifest.components is required.`
      );
    }
    const files = [
      components.kronecker_sum?.shard_file,
      components.coordinate_inr?.shard_file,
      components.sparse_outliers?.shard_file,
    ];
    if (files.some((file) => typeof file !== 'string' || file.trim().length === 0)) {
      throw new Error(
        `[DopplerLoader] FUNCTIONAL_DESCRIPTOR tensor "${name}" must declare kronecker, SIREN, and sparse shard_file values.`
      );
    }
    return files.map((file) => file.trim());
  }

export async function _loadDescriptorShardFile(file, name) {
    const loadAuxiliaryFile = this._loadAuxiliaryFile;
    const payload = loadAuxiliaryFile
      ? await loadAuxiliaryFile(file)
      : await loadAuxFile(file);
    if (payload == null) {
      throw new Error(
        `[DopplerLoader] Descriptor shard "${file}" for tensor "${name}" was not found.`
      );
    }
    if (payload instanceof Uint8Array) {
      return payload;
    }
    if (ArrayBuffer.isView(payload)) {
      return new Uint8Array(payload.buffer, payload.byteOffset, payload.byteLength);
    }
    if (payload instanceof ArrayBuffer) {
      return new Uint8Array(payload);
    }
    throw new Error(
      `[DopplerLoader] Descriptor shard "${file}" for tensor "${name}" must load as ArrayBuffer or Uint8Array.`
    );
  }

export async function _assertDescriptorHash(location, name, descriptorShards) {
    const descriptorHash = location?.descriptorManifest?.descriptor_hash;
    if (typeof descriptorHash !== 'string' || !descriptorHash.trim()) {
      return;
    }
    const match = /^sha256:([a-f0-9]{64})$/i.exec(descriptorHash.trim());
    if (!match) {
      throw new Error(
        `[DopplerLoader] FUNCTIONAL_DESCRIPTOR tensor "${name}" descriptor_hash must be sha256:<64 hex chars>.`
      );
    }
    const totalBytes = Array.from(descriptorShards.values()).reduce((sum, bytes) => sum + bytes.byteLength, 0);
    const combined = new Uint8Array(totalBytes);
    let offset = 0;
    for (const bytes of descriptorShards.values()) {
      combined.set(bytes, offset);
      offset += bytes.byteLength;
    }
    const actual = await computeHash(combined, 'sha256');
    if (actual.toLowerCase() !== match[1].toLowerCase()) {
      throw new Error(
        `[DopplerLoader] FUNCTIONAL_DESCRIPTOR tensor "${name}" descriptor hash mismatch. ` +
        `Expected ${descriptorHash}, got sha256:${actual}.`
      );
    }
  }

export async function _assembleFunctionalDescriptorData(location, name) {
    const shardFiles = this._getDescriptorShardFiles(location, name);
    const descriptorShards = new Map();
    for (const file of shardFiles) {
      descriptorShards.set(file, await this._loadDescriptorShardFile(file, name));
    }
    await this._assertDescriptorHash(location, name, descriptorShards);
    const data = new Uint8Array(0);
    Object.defineProperty(data, 'descriptorShards', {
      value: descriptorShards,
      enumerable: false,
    });
    return data;
  }

export async function _assembleShardData(location, name, options = {}) {
    const loadShard = this._getLoadShard();
    const loadShardRange = (idx, offset, length) => this.shardCache.loadRange(idx, offset, length);
    const data = await assembleShardData(location, name, loadShard, loadShardRange, options);
    const companions = Array.isArray(location?.storage?.companions)
      ? location.storage.companions
      : [];
    if (companions.length === 0) {
      return data;
    }
    const storageCompanions = {};
    for (const companion of companions) {
      const companionLocation = this.tensorLocations.get(companion.tensorId);
      if (!companionLocation) {
        throw new Error(
          `[DopplerLoader] Tensor "${name}" storage companion "${companion.tensorId}" for role "${companion.role}" was not found.`
        );
      }
      storageCompanions[companion.role] = {
        tensorId: companion.tensorId,
        location: companionLocation,
        bytes: await assembleShardData(companionLocation, companion.tensorId, loadShard, loadShardRange),
      };
    }
    Object.defineProperty(data, 'storageCompanions', {
      value: storageCompanions,
      enumerable: false,
    });
    return data;
  }

export function _shouldStreamUploadToGPU(location) {
    if (this._isFunctionalDescriptorLocation(location)) return false;
    if (!location?.size || location.size <= 0) return false;
    if (hasSourceTransform(location)) return false;
    if (Array.isArray(location?.storage?.companions) && location.storage.companions.length > 0) return false;
    if (requiresCpuF16ToF32MatmulMaterialization(location, this.gpuCapabilities, this.keepF32Weights)) return false;
    if (this.shardCache.hasCustomLoader && !this.shardCache.canStreamRanges) return false;
    const chunkBytes = this._loadingConfig?.storage?.backend?.streaming?.readChunkBytes ?? 0;
    if (!Number.isFinite(chunkBytes) || chunkBytes <= 0) return false;
    // Always stream multi-span tensors to avoid loading whole shards + assembling on CPU.
    if (location.spans && location.spans.length > 0) {
      return true;
    }
    // Conservative default: only stream "large" single-span tensors to avoid turning
    // OPFS into many small random reads that can be slower than whole-shard caching.
    const minStreamBytes = Math.max(16 * 1024 * 1024, chunkBytes * 4);
    return location.size >= minStreamBytes;
  }

export async function _assembleShardDataToGpuBuffer(location, name) {
    const device = getDevice();
    if (!device) {
      throw new Error('GPU device not available');
    }
    const rawChunkBytes = Number(this._loadingConfig?.storage?.backend?.streaming?.readChunkBytes);
    const chunkBytes = Number.isFinite(rawChunkBytes) && rawChunkBytes > 0
      ? Math.floor(rawChunkBytes)
      : 1;

    // queue.writeBuffer requires 4-byte aligned sizes; we pad the buffer.
    const alignedSize = Math.ceil(location.size / 4) * 4;
    const raw = acquireBuffer(alignedSize, undefined, `raw_${name}`);
    let complete = false;

    try {
      let dstOffset = 0;
      let pendingBytes = null;
      const writeAlignedChunk = (bytes) => {
        if (bytes.byteLength === 0) return;
        device.queue.writeBuffer(raw, dstOffset, bytes, 0, bytes.byteLength);
        dstOffset += bytes.byteLength;
      };
      const uploadChunk = (bytes) => {
        let merged = bytes;
        if (pendingBytes && pendingBytes.byteLength > 0) {
          merged = new Uint8Array(pendingBytes.byteLength + bytes.byteLength);
          merged.set(pendingBytes, 0);
          merged.set(bytes, pendingBytes.byteLength);
          pendingBytes = null;
        }
        const alignedLength = merged.byteLength - (merged.byteLength % 4);
        if (alignedLength > 0) {
          writeAlignedChunk(merged.subarray(0, alignedLength));
        }
        const remainder = merged.byteLength - alignedLength;
        pendingBytes = remainder > 0 ? merged.slice(alignedLength) : null;
      };
      const streamRange = (idx, offset, length) => this.shardCache.streamRange(idx, offset, length, { chunkBytes });

      if (location.spans) {
        for (const span of location.spans) {
          for await (const chunk of streamRange(span.shardIndex, span.offset, span.size)) {
            uploadChunk(chunk);
          }
        }
      } else {
        for await (const chunk of streamRange(location.shardIndex, location.offset, location.size)) {
          uploadChunk(chunk);
        }
      }

      if (pendingBytes && pendingBytes.byteLength > 0) {
        const padded = new Uint8Array(4);
        padded.set(pendingBytes, 0);
        writeAlignedChunk(padded);
        dstOffset -= (4 - pendingBytes.byteLength);
        pendingBytes = null;
      }

      if (dstOffset !== location.size) {
        throw new Error(
          `Stream upload short read for "${name}": got=${dstOffset}, expected=${location.size}.`
        );
      }
      complete = true;
      return raw;
    } finally {
      if (!complete) {
        releaseBuffer(raw);
      }
    }
  }

export async function _loadEmbeddings(_onProgress) {
    
    const ctx = {
      tensorLocations: this.tensorLocations,
      loadTensor: (name, toGPU, silent) => this._loadTensor(name, toGPU, silent),
      loadShardRange: (index, offset, length) => this.shardCache.loadRange(index, offset, length),
      shouldStreamLargeWeight: (name, loc, label) => this._shouldStreamLargeWeight(name, loc, label),
      resolveWeightLayout: (loc) => this._resolveWeightLayout(loc),
      gpuBuffers: this.gpuBuffers,
      keepF32Weights: this.keepF32Weights,
      // Keep embedding weights in F32 when manifest quantization requires it.
      // gather.wgsl reads embeddings as f32; downcasting here corrupts reads.
      preserveF32Embeddings: String(this.manifest?.quantizationInfo?.embeddings ?? '').toLowerCase() === 'f32',
      hostHasShaderF16: this.gpuCapabilities?.hasF16 ?? null,
      embeddingKernel: this.manifest?.inference?.execution?.kernels?.embed ?? null,
    };

    this.embeddings = await loadEmbeddings(ctx);
    this.perLayerInputWeights = await loadPerLayerInputWeights({
      modelId: this.manifest?.modelId ?? null,
      tensorLocations: this.tensorLocations,
      gpuBuffers: this.gpuBuffers,
      loadTensor: (name, toGPU, silent) => this._loadTensor(name, toGPU, silent),
      shouldStreamLargeWeight: (name, loc, label) => this._shouldStreamLargeWeight(name, loc, label),
      loadShardRange: (index, offset, length) => this.shardCache.loadRange(index, offset, length),
      resolveWeightLayout: (loc) => this._resolveWeightLayout(loc),
      perLayerInputSession: this._perLayerInputSession,
    }, this.manifest?.architecture ?? null);
  }

export async function _loadLayer(layerIdx, _onProgress) {
    const textConfig = (
      this.manifest?.config?.text_config
      && typeof this.manifest.config.text_config === 'object'
      && !Array.isArray(this.manifest.config.text_config)
    )
      ? this.manifest.config.text_config
      : this.manifest?.config ?? null;

    
    const ctx = {
      tensorLocations: this.tensorLocations,
      loadTensor: (name, toGPU, silent) => this._loadTensor(name, toGPU, silent),
      needsNormWeightOffset: () => this._needsNormWeightOffset(),
      gpuBuffers: this.gpuBuffers,
      keepF32Weights: this.keepF32Weights,
      isMoE: this.isMoE,
      isExpertLayer: (idx) => this._isExpertLayer(idx),
      loadDenseFfnForMoeLayers: this.manifest?.inference?.ffn?.branchMode === 'dense_plus_moe',
      numHeads: this.manifest?.architecture?.numAttentionHeads ?? null,
      numKVHeads: this.manifest?.architecture?.numKeyValueHeads ?? null,
      headDim: this.manifest?.architecture?.headDim ?? null,
      hiddenSize: this.manifest?.architecture?.hiddenSize ?? null,
      linearNumKeyHeads: textConfig?.linear_num_key_heads ?? this.manifest?.architecture?.linearNumKeyHeads ?? null,
      linearNumValueHeads: textConfig?.linear_num_value_heads ?? this.manifest?.architecture?.linearNumValueHeads ?? null,
      linearKeyHeadDim: textConfig?.linear_key_head_dim ?? this.manifest?.architecture?.linearKeyHeadDim ?? null,
      linearValueHeadDim: textConfig?.linear_value_head_dim ?? this.manifest?.architecture?.linearValueHeadDim ?? null,
    };

    const weights = await loadLayer(ctx, layerIdx);
    this.layers.set(layerIdx, weights);
  }

export async function _loadFinalWeights(_onProgress) {
    const tieWordEmbeddings = this.manifest?.inference?.output?.tieWordEmbeddings;
    if (tieWordEmbeddings == null) {
      const modelId = this.manifest?.modelId ?? 'unknown';
      throw new Error(
        `Manifest "${modelId}" is missing inference.output.tieWordEmbeddings. ` +
        'Re-convert the model with a complete manifest.inference config.'
      );
    }

    
    const ctx = {
      tensorLocations: this.tensorLocations,
      loadTensor: (name, toGPU, silent) => this._loadTensor(name, toGPU, silent),
      loadShardRange: (index, offset, length) => this.shardCache.loadRange(index, offset, length),
      needsNormWeightOffset: () => this._needsNormWeightOffset(),
      shouldStreamLargeWeight: (name, loc, label) => this._shouldStreamLargeWeight(name, loc, label),
      resolveWeightLayout: (loc) => this._resolveWeightLayout(loc),
      embeddings: this.embeddings,
      embeddingPostprocessor: this.manifest?.inference?.output?.embeddingPostprocessor ?? null,
      finalNormBiasTensor: this.manifest?.inference?.normalization?.finalNormBiasTensor ?? null,
      lmHeadBiasTensor: this.manifest?.inference?.output?.lmHeadBiasTensor ?? null,
      diffusionGemmaSelfConditioning: this.manifest?.inference?.diffusionGemma?.selfConditioning === true,
      modelType: this.manifest?.modelType ?? null,
      tieWordEmbeddings,
      gpuBuffers: this.gpuBuffers,
      keepF32Weights: this.keepF32Weights,
      normOffsetDebugLogged: this._normOffsetDebugLogged,
    };

    const result = await loadFinalWeights(ctx);
    this.finalNorm = result.finalNorm;
    this.finalNormBias = result.finalNormBias;
    this.lmHead = result.lmHead;
    this.lmHeadBias = result.lmHeadBias;
    this.embeddingPostprocessor = result.embeddingPostprocessor;
    this.diffusionGemmaSelfConditioning = result.diffusionGemmaSelfConditioning;
    this._normOffsetDebugLogged = result.normOffsetDebugLogged;
  }
