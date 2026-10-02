import { openModelStore, verifyIntegrity, loadManifestFromStore } from '../storage/shard-manager.js';
import { parseManifest } from '../formats/rdrr/index.js';
import { getDevice } from '../gpu/device.js';
import { formatBytes } from '../storage/quota.js';
import { log } from '../debug/index.js';
import { validateManifestInference } from '../config/schema/index.js';
import { detectMoE } from './model-load-validation.js';
import { createLoadTiming, finishLoadPhase, finishLoadTiming, nowMs, roundLoadTimingMs } from './load-timing.js';
import { resolveLayerPartition } from '../inference/pipelines/text/layer-partition-contract.js';
import { requiresTiedEmbeddingLoad } from './final-weights-loader.js';

/** @type {import('./model-load.js').load} */
export async function load(modelId, options) {
  const started = nowMs();
  const preservedManifest = this.shardCache.hasCustomLoader ? this.manifest : null;
  // Capture placement before the first asynchronous storage or device operation.
  const snapshot = { ...options, partition: options.partition == null ? null : structuredClone(options.partition) };
  try {
    return await loadModel.call(this, modelId, snapshot);
  } catch (error) {
    if (this.loadTiming?.status === 'running') {
      finishLoadTiming(this.loadTiming, 'failed', started, error);
    }
    try {
      if (this._memoryMonitor) this._stopMemoryLogging('failed');
      await this.unload();
    } catch (cleanupError) {
      throw new AggregateError([error, cleanupError], 'Model load and cleanup failed.', { cause: error });
    } finally {
      if (preservedManifest) this.manifest = preservedManifest;
    }
    throw error;
  }
}

/** @type {import('./model-load.js').load} */
async function loadModel(modelId, options) {
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
    const loadTiming = createLoadTiming(modelId, this.shardCache.hasCustomLoader);
    this.loadTiming = loadTiming;

    this._startMemoryLogging();
    this._assertResidentBudget('load start');

    if (!this.shardCache.hasCustomLoader) {
      await openModelStore(modelId);
      const manifestJson = await loadManifestFromStore();
      if (manifestJson == null) throw new Error(`No manifest available for ${modelId}.`);
      this.manifest = parseManifest(manifestJson);
    }

    if (!this.manifest) {
      throw new Error('No manifest available. Set manifest via setManifest() or ensure OPFS has the model.');
    }

    validateManifestInference(this.manifest);
    const partition = resolveLayerPartition(this.manifest, options.partition);

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
    loadTiming.totalBytes = totalBytes;
    loadTiming.totalShards = totalShards;
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
      loadTiming.bytesLoaded = bytesLoaded;
      loadTiming.shardsLoaded = shardsLoaded;
    };

    
    /** @type {(stage: import('./loader-types.js').LoadProgress['stage'], baseProgress: number, detail: string) => void} */
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
    const originalLoadShard = this._loadShard.bind(this);

    
    this._loadShardOverride = async (shardIndex, options) => {
      const shardInfo = this.manifest?.shards?.[shardIndex];
      const shardSize = shardInfo?.size || 0;
      const shardName = shardInfo?.filename ?? `index=${shardIndex}`;
      let data;
      try {
        data = await originalLoadShard(shardIndex, options);
      } catch (error) {
        const modelId = this.manifest?.modelId ?? 'unknown';
        const shardUrl = shardInfo && ('url' in shardInfo ? shardInfo.url
          : 'path' in shardInfo ? shardInfo.path : 'unknown');
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
        loadTiming.bytesLoaded = bytesLoaded;
        loadTiming.shardsLoaded = shardsLoaded;
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
      // A owns input embeddings. A tied output head is an explicit shared weight
      // dependency of B; neither group loads the other group's decoder layers.
      if (!partition || partition.hasEmbedding
        || partition.hasLmHead && requiresTiedEmbeddingLoad(this.tensorLocations, this.manifest.inference.output.tieWordEmbeddings)) {
        await this._loadEmbeddings(onProgress);
      }
      finishLoadPhase(this.loadTiming, activeLoadPhase, phaseStart);
      this._assertResidentBudget('embeddings');

      /** @type {(value: unknown) => number} */
      const resolveNumLayers = (value) => {
        const normalized = Number(value);
        if (!Number.isInteger(normalized) || normalized <= 0) {
          return 0;
        }
        return normalized;
      };

      const manifestConfig = this.manifest.config;
      const textConfig = manifestConfig?.text_config;
      const architecture = this.manifest.architecture;
      const architectureLayers = typeof architecture === 'object' ? architecture?.numLayers : undefined;
      const layerCountCandidates = [
        manifestConfig?.num_hidden_layers,
        manifestConfig?.blockCount,
        textConfig && typeof textConfig === 'object' && 'num_hidden_layers' in textConfig
          ? textConfig.num_hidden_layers : undefined,
        manifestConfig?.n_layer,
        architectureLayers,
      ];
      const numLayers = layerCountCandidates
        .map(resolveNumLayers)
        .find((count) => Number.isInteger(count) && count > 0);

      if (numLayers === undefined || !Number.isInteger(numLayers)) {
        throw new Error(
          `Manifest "${this.manifest.modelId ?? 'unknown'}" missing or invalid layer count. ` +
          `Expected one of manifest.config.num_hidden_layers/blockCount/text_config.num_hidden_layers/n_layer ` +
          `or manifest.architecture.numLayers.`
        );
      }

      if (partition && numLayers !== architectureLayers) {
        throw new Error('Partition layer count conflicts with the manifest config.');
      }
      const firstLayer = partition?.layerRange[0] ?? 0;
      const lastLayer = partition?.layerRange[1] ?? numLayers - 1;
      const selectedLayerCount = lastLayer - firstLayer + 1;
      log.info('Loader', `Layers: ${firstLayer}-${lastLayer}`);

      inLayerPhase = true;
      activeLoadPhase = 'layers';
      const layersStartTime = performance.now();
      let layerTotalMs = 0;
      let maxLayerMs = 0;
      let maxLayer = null;

      for (let l = firstLayer; l <= lastLayer; l++) {
        const layerStart = performance.now();
        const layerPromise = this._loadLayer(l, onProgress);
        if (!partition) this._prefetchLayerShards(l);
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
          const completedLayers = l - firstLayer + 1;
          const layerFraction = completedLayers / selectedLayerCount;
          const layerProgress = 0.80 + layerFraction * 0.05;
          onProgress({
            stage: 'layers',
            layer: completedLayers,
            total: selectedLayerCount,
            progress: layerProgress,
            shard: shardsLoaded,
            totalShards,
            bytesLoaded,
            totalBytes,
            bytesPerSecond: 0,
            message: `Layer ${l}: ${completedLayers}/${selectedLayerCount} assigned layers`,
          });
        }
        this._assertResidentBudget(`layer ${l + 1}`);
      }

      const layersTotalTime = ((performance.now() - layersStartTime) / 1000).toFixed(2);
      loadTiming.layers = {
        count: selectedLayerCount,
        totalMs: roundLoadTimingMs(layerTotalMs),
        meanMs: roundLoadTimingMs(layerTotalMs / selectedLayerCount),
        maxMs: maxLayer == null ? null : roundLoadTimingMs(maxLayerMs),
        maxLayer,
      };
      finishLoadPhase(this.loadTiming, activeLoadPhase, layersStartTime);
      log.info('Loader', `Layers: ${selectedLayerCount} complete (${layersTotalTime}s)`);

      reportProgress('gpu_transfer', 0.85, 'Loading final weights...');
      activeLoadPhase = 'finalWeights';
      phaseStart = nowMs();
      if (!partition || partition.hasLmHead) await this._loadFinalWeights(onProgress);
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
      throw loadError;
    }
    return  (this.manifest?.config) || {};
  }
