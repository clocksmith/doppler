import { getMemoryCapabilities } from '../memory/capability.js';
import { detectUnifiedMemory } from '../memory/unified-detect.js';
import { getHeapManager } from '../memory/heap-manager.js';
import {
  initStorage,
  openModelStore,
  verifyIntegrity,
  loadManifestFromStore,
  loadAuxFile,
  computeHash,
} from '../storage/shard-manager.js';
import { clearManifest, parseManifest, setManifest as setCurrentManifest } from '../formats/rdrr/index.js';
import { initDevice, getDevice, getKernelCapabilities } from '../gpu/device.js';
import {
  PersistentBufferSet,
  acquireBuffer,
  isBufferActive,
  releaseBuffer,
  forceBufferPoolReclaim,
} from '../memory/buffer-pool.js';
import { getExpertCache } from './experts/expert-cache.js';
import { formatBytes } from '../storage/quota.js';
import { log, trace as debugTrace } from '../debug/index.js';
import { isGpuBufferInstance, isWeightBuffer } from '../gpu/weight-buffer.js';
import { createShardCache } from './shard-cache.js';
import { validateManifestInference } from '../config/schema/index.js';
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
import { detectMoE } from './model-load-validation.js';
import {
  createLoadTiming,
  finishLoadPhase,
  finishLoadTiming,
  nowMs,
  roundLoadTimingMs,
} from './load-timing.js';
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

export async function load(modelId, options = {}) {
    const { onProgress = null, verifyHashes } = options;
    if (verifyHashes == null) {
      throw new Error('Loader.load requires explicit verifyHashes (runtime.loading.shardCache.verifyHashes).');
    }

    if (!this.heapManager) {
      await this.init();
    }

    // Check order matters: isLoaded is the fast-path indicator; modelId catches
    // partial loads that set the ID before completing; tensorLocations/shardCache
    // detect interrupted builds; layers/experts/gpuBuffers catch residual GPU
    // state from a prior model that was never fully unloaded.
    const hasExistingModelState =
      this.isLoaded ||
      this.modelId !== null ||
      this.tensorLocations.size > 0 ||
      this.shardCache.size > 0 ||
      this.layers.size > 0 ||
      this.experts.size > 0 ||
      this.gpuBuffers.size > 0;

    const preservedManifest = this.shardCache.hasCustomLoader ? this.manifest : null;

    if (hasExistingModelState) {
      await this.unload();
    }

    if (preservedManifest) {
      this.manifest = preservedManifest;
    }

    log.info('Loader', `Loading: ${modelId}`);
    this.modelId = modelId;
    const loadTimingStart = nowMs();
    let activeLoadPhase = 'preflight';
    let phaseStart = loadTimingStart;
    this.loadTiming = createLoadTiming(modelId, this.shardCache.hasCustomLoader);

    this._startMemoryLogging();
    this._assertResidentBudget('load start');

    if (!this.shardCache.hasCustomLoader) {
      await openModelStore(modelId);
      const manifestJson = await loadManifestFromStore();
      this.manifest = parseManifest(manifestJson);
    }

    if (!this.manifest) {
      throw new Error('No manifest available. Set manifest via setManifest() or ensure OPFS has the model.');
    }

    validateManifestInference(this.manifest);

    this.isMoE = detectMoE(this.manifest);

    this.shardCache.configureForModel(this.manifest, this.shardCache.hasCustomLoader);

    if (!this.isMoE && !this.isUnifiedMemory) {
      log.warn('Loader', 'Dense model on discrete GPU - performance limited. Consider MoE model.');
    }

    if (!this.shardCache.hasCustomLoader) {
      const integrity = await verifyIntegrity({ checkHashes: false });
      if (!integrity.valid) {
        throw new Error(
          `Artifact contract preflight failed for "${this.manifest?.modelId ?? modelId}". ` +
          `Missing shards: ${integrity.missingShards.length}, ` +
          `corrupt shards: ${integrity.corruptShards.length}. ` +
          'Re-import, re-download, or provide a manifest with a valid weightsRef.'
        );
      }
    }

    const totalBytes = (this.manifest.shards || []).reduce((sum, s) => sum + (s.size || 0), 0);
    const totalShards = this.manifest.shards?.length || 0;
    this.loadTiming.totalBytes = totalBytes;
    this.loadTiming.totalShards = totalShards;
    finishLoadPhase(this.loadTiming, activeLoadPhase, phaseStart);

    activeLoadPhase = 'tensorLocations';
    phaseStart = nowMs();
    await this._buildTensorLocations();
    finishLoadPhase(this.loadTiming, activeLoadPhase, phaseStart);

    const loadStartTime = Date.now();
    let bytesLoaded = 0;
    let shardsLoaded = 0;
    this.shardCache.resetCustomReadStats();

    const syncCustomReadStats = () => {
      if (!this.shardCache.hasCustomLoader) return;
      const stats = this.shardCache.customReadStats;
      bytesLoaded = stats.bytesRead;
      shardsLoaded = stats.shardsRead;
      this.loadTiming.bytesLoaded = bytesLoaded;
      this.loadTiming.shardsLoaded = shardsLoaded;
    };

    
    const reportProgress = (stage, baseProgress, detail) => {
      if (!onProgress || typeof onProgress !== 'function') return;
      syncCustomReadStats();
      const elapsed = (Date.now() - loadStartTime) / 1000;
      const speed = elapsed > 0 ? bytesLoaded / elapsed : 0;
      const speedStr = speed > 0 ? `${formatBytes(speed)}/s` : '';
      const message = detail ||
        `${formatBytes(bytesLoaded)} / ${formatBytes(totalBytes)} ${speedStr ? `- ${speedStr}` : ''}`;
      onProgress({
        stage,
        progress: baseProgress,
        shard: shardsLoaded,
        totalShards,
        bytesLoaded,
        totalBytes,
        bytesPerSecond: speed,
        message,
      });
    };

    if (onProgress) {
      onProgress({ stage: 'manifest', progress: 0.05, message: 'Parsing manifest...' });
    }

    
    const loadedShardIndices = new Set();
    let inLayerPhase = false;
    const originalLoadShard = (shardIndex, options) => this._loadShard(shardIndex, options);

    
    this._loadShardOverride = async (shardIndex, options) => {
      const shardInfo = this.manifest?.shards?.[shardIndex];
      const shardSize = shardInfo?.size || 0;
      const shardName = shardInfo?.filename ?? `index=${shardIndex}`;
      let data;
      try {
        data = await originalLoadShard(shardIndex, options);
      } catch (error) {
        const modelId = this.manifest?.modelId ?? 'unknown';
        const shardUrl = shardInfo?.url ?? shardInfo?.path ?? 'unknown';
        const sizeStr = shardSize > 0 ? `, size=${formatBytes(shardSize)}` : '';
        log.error(
          'Loader',
          `Failed to load shard ${shardIndex}/${totalShards} "${shardName}" ` +
          `for model "${modelId}" (url=${shardUrl}${sizeStr}): ${error.message}`
        );
        throw error;
      }

      if (!loadedShardIndices.has(shardIndex)) {
        loadedShardIndices.add(shardIndex);
        bytesLoaded += shardSize;
        shardsLoaded++;
        this.loadTiming.bytesLoaded = bytesLoaded;
        this.loadTiming.shardsLoaded = shardsLoaded;
        if (!inLayerPhase) {
          const pct = 0.1 + Math.min(bytesLoaded / totalBytes, 1.0) * 0.7;
          const elapsed = (Date.now() - loadStartTime) / 1000;
          const speed = elapsed > 0 ? bytesLoaded / elapsed : 0;
          const sourceInfo = this.shardCache.lastSource;
          const sourceStr = sourceInfo
            ? [sourceInfo.source, sourceInfo.mode, sourceInfo.path].filter(Boolean).join('/')
            : 'unknown';
          const fallbackStr = sourceInfo?.fallback && sourceInfo.fallback !== 'none'
            ? ` fallback=${sourceInfo.fallback}`
            : '';
          const elapsedStr = sourceInfo && sourceInfo.elapsed > 0 ? ` ${sourceInfo.elapsed.toFixed(2)}s` : '';
          if (onProgress) {
            onProgress({
              stage: 'shards',
              progress: pct,
              shard: shardsLoaded,
              totalShards,
              bytesLoaded,
              totalBytes,
              bytesPerSecond: speed,
              message: `Shard ${shardIndex}: ${sourceStr} (${formatBytes(shardSize)}${elapsedStr}${fallbackStr})`,
            });
          }
        }
      }
      return data;
    };

    
    let loadError = null;
    try {
      reportProgress('shards', 0.1, 'Loading embeddings...');
      activeLoadPhase = 'embeddings';
      phaseStart = nowMs();
      await this._loadEmbeddings(onProgress);
      finishLoadPhase(this.loadTiming, activeLoadPhase, phaseStart);
      this._assertResidentBudget('embeddings');

      const resolveNumLayers = (value) => {
        const normalized = Number(value);
        if (!Number.isInteger(normalized) || normalized <= 0) {
          return 0;
        }
        return normalized;
      };

      const manifestConfig = this.manifest.config;
      const layerCountCandidates = [
        manifestConfig?.num_hidden_layers,
        manifestConfig?.blockCount,
        manifestConfig?.text_config?.num_hidden_layers,
        manifestConfig?.n_layer,
        this.manifest.architecture?.numLayers,
      ];
      const numLayers = layerCountCandidates
        .map(resolveNumLayers)
        .find((count) => Number.isInteger(count) && count > 0);

      if (!Number.isInteger(numLayers)) {
        throw new Error(
          `Manifest "${this.manifest.modelId ?? 'unknown'}" missing or invalid layer count. ` +
          `Expected one of manifest.config.num_hidden_layers/blockCount/text_config.num_hidden_layers/n_layer ` +
          `or manifest.architecture.numLayers.`
        );
      }

      log.info('Loader', `Layers: 0-${numLayers - 1}`);

      inLayerPhase = true;
      activeLoadPhase = 'layers';
      const layersStartTime = performance.now();
      let layerTotalMs = 0;
      let maxLayerMs = 0;
      let maxLayer = null;

      for (let l = 0; l < numLayers; l++) {
        const layerStart = performance.now();
        const layerPromise = this._loadLayer(l, onProgress);
        this._prefetchLayerShards(l);
        await layerPromise;
        const layerElapsedMs = performance.now() - layerStart;
        layerTotalMs += layerElapsedMs;
        if (layerElapsedMs > maxLayerMs) {
          maxLayerMs = layerElapsedMs;
          maxLayer = l;
        }
        const layerElapsed = (layerElapsedMs / 1000).toFixed(2);
        log.verbose('Loader', `  Layer ${l}: ${layerElapsed}s`);

        await new Promise(r => setTimeout(r, 0));

        const { flushIntervalLayers, flushThresholdBytes, gpuQueueFlushLayers } = this._loadingConfig.memoryManagement;
        const cacheBytes = this.shardCache.totalBytes;
        const shouldFlushCache = !this.shardCache.hasCustomLoader && l > 0 && (l % flushIntervalLayers === 0 || cacheBytes > flushThresholdBytes);
        if (shouldFlushCache) {
          this.shardCache.clear();
        }
        if (l > 0 && l % gpuQueueFlushLayers === 0) {
          const device = getDevice();
          if (device) {
            await device.queue.onSubmittedWorkDone();
          }
        }

        if (onProgress) {
          syncCustomReadStats();
          const layerFraction = (l + 1) / numLayers;
          const layerProgress = 0.80 + layerFraction * 0.05;
          onProgress({
            stage: 'layers',
            layer: l + 1,
            total: numLayers,
            progress: layerProgress,
            shard: shardsLoaded,
            totalShards,
            bytesLoaded,
            totalBytes,
            bytesPerSecond: 0,
            message: `Layer ${l + 1}/${numLayers}`,
          });
        }
        this._assertResidentBudget(`layer ${l + 1}`);
      }

      const layersTotalTime = ((performance.now() - layersStartTime) / 1000).toFixed(2);
      this.loadTiming.layers = {
        count: numLayers,
        totalMs: roundLoadTimingMs(layerTotalMs),
        meanMs: numLayers > 0 ? roundLoadTimingMs(layerTotalMs / numLayers) : null,
        maxMs: maxLayer == null ? null : roundLoadTimingMs(maxLayerMs),
        maxLayer,
      };
      finishLoadPhase(this.loadTiming, activeLoadPhase, layersStartTime);
      log.info('Loader', `Layers: ${numLayers} complete (${layersTotalTime}s)`);

      reportProgress('gpu_transfer', 0.85, 'Loading final weights...');
      activeLoadPhase = 'finalWeights';
      phaseStart = nowMs();
      await this._loadFinalWeights(onProgress);
      finishLoadPhase(this.loadTiming, activeLoadPhase, phaseStart);
      this._assertResidentBudget('final weights');
      syncCustomReadStats();

      if (onProgress) {
        onProgress({
          stage: 'complete',
          progress: 1.0,
          shard: shardsLoaded,
          totalShards,
          bytesLoaded,
          totalBytes,
        });
      }

      this.isLoaded = true;
      const totalTime = ((Date.now() - loadStartTime) / 1000).toFixed(2);
      const avgSpeed = formatBytes(bytesLoaded / (Date.now() - loadStartTime) * 1000);
      log.info('Loader', `Complete: ${formatBytes(bytesLoaded)} in ${totalTime}s (${avgSpeed}/s)`);

      activeLoadPhase = 'cleanup';
      phaseStart = nowMs();
      this.shardCache.clear();
      finishLoadPhase(this.loadTiming, activeLoadPhase, phaseStart);
      finishLoadTiming(this.loadTiming, 'complete', loadTimingStart);

      return  (this.manifest.config) || {};
    } catch (error) {
      loadError = error;
      syncCustomReadStats();
      finishLoadTiming(this.loadTiming, 'failed', loadTimingStart, error, activeLoadPhase);
    } finally {
      this._loadShardOverride = null;
      if (this._memoryMonitor) {
        this._stopMemoryLogging(loadError ? 'failed' : 'complete');
      }
    }

    if (loadError) {
      await this.unload();
      if (preservedManifest) {
        this.manifest = preservedManifest;
      }
      throw loadError;
    }
    return  (this.manifest?.config) || {};
  }

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
