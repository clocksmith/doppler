

import { log } from '../src/debug/index.js';
import { createPipeline } from '../src/inference/pipeline.js';
import {
  openModelStore,
  loadManifestFromStore,
  loadShard,
} from '../src/storage/shard-manager.js';
import { parseManifest } from '../src/storage/rdrr-format.js';
import { getMemoryCapabilities } from '../src/memory/capability.js';
import { getHeapManager } from '../src/memory/heap-manager.js';
import { initDevice, getKernelCapabilities, getDevice } from '../src/gpu/device.js';
import { destroyBufferPool } from '../src/memory/buffer-pool.js';

export class ModelLoader {
  #pipeline = null;

  #currentModel = null;

  #memoryCapabilities = null;

  #callbacks;

  constructor(callbacks = {}) {
    this.#callbacks = callbacks;
  }

  get pipeline() {
    return this.#pipeline;
  }

  get currentModel() {
    return this.#currentModel;
  }

  get memoryCapabilities() {
    return this.#memoryCapabilities;
  }

  async load(model, options = {}) {
    const sources = model.sources || {};
    const hasServer = !!sources.server;
    const hasBrowser = !!sources.browser;

    if (!hasServer && !hasBrowser) {
      throw new Error('Model not available locally. Download it first.');
    }

    // Determine source
    let useServer;
    if (options.preferredSource === 'server' && hasServer) {
      useServer = true;
    } else if (options.preferredSource === 'browser' && hasBrowser) {
      useServer = false;
    } else {
      useServer = hasServer;
    }

    const sourceInfo = useServer ? sources.server : sources.browser;
    const sourceType = useServer ? 'server' : 'browser';

    log.info('ModelLoader', `Loading model: ${model.name} from ${sourceType}`);
    this.#callbacks.onProgress?.({ phase: 'source', percent: 0, message: 'Starting...' });

    // Unload existing pipeline
    await this.unload();

    let manifest;
    let loadShardFn;

    // Determine load source type for progress UI
    const isLocalServer = sourceInfo.url?.match(/^(https?:\/\/)?(localhost|127\.0\.0\.1|0\.0\.0\.0|file:)/i);
    const loadSourceType = useServer ? (isLocalServer ? 'disk' : 'network') : 'cache';
    this.#callbacks.onSourceType?.(loadSourceType);

    if (useServer) {
      // Load from HTTP
      this.#callbacks.onProgress?.({ phase: 'source', percent: 5, message: 'Fetching manifest...' });
      const manifestUrl = `${sourceInfo.url}/manifest.json`;
      const response = await fetch(manifestUrl);
      if (!response.ok) throw new Error(`Failed to fetch manifest: ${response.status}`);
      manifest = parseManifest(await response.text());

      loadShardFn = async (idx) => {
        const shard = manifest.shards[idx];
        const shardUrl = `${sourceInfo.url}/${shard.filename}`;
        const res = await fetch(shardUrl);
        if (!res.ok) throw new Error(`Failed to fetch shard ${idx}: ${res.status}`);
        return await res.arrayBuffer();
      };
    } else {
      // Load from OPFS
      await openModelStore(sourceInfo.id);
      this.#callbacks.onProgress?.({ phase: 'source', percent: 5, message: 'Loading manifest...' });
      const manifestJson = await loadManifestFromStore();
      manifest = parseManifest(manifestJson);
      loadShardFn = (idx) => loadShard(idx);
    }

    // Initialize GPU
    this.#callbacks.onProgress?.({ phase: 'gpu', percent: 5, message: 'Initializing...' });

    const device = getDevice() || (await initDevice());
    const gpuCaps = getKernelCapabilities();
    const memCaps = await getMemoryCapabilities();
    this.#memoryCapabilities = memCaps;

    const heapManager = getHeapManager();
    await heapManager.init();

    this.#callbacks.onProgress?.({ phase: 'gpu', percent: 10, message: 'Creating pipeline...' });

    // Create pipeline
    this.#pipeline = await createPipeline(manifest, {
      gpu: {
        capabilities: gpuCaps,
        device: device,
      },
      memory: {
        capabilities: memCaps,
        heapManager: heapManager,
      },
      storage: {
        loadShard: loadShardFn,
      },
      baseUrl: useServer ? sourceInfo.url : undefined,
      onProgress: (progress) => {
        this.#handlePipelineProgress(progress);
      },
    });

    this.#currentModel = model;
    log.info('ModelLoader', `Model loaded: ${model.name} (${model.key})`);

    return this.#pipeline;
  }

  #handlePipelineProgress(progress) {
    const stage = progress.stage || 'layers';

    if (stage === 'manifest' || stage === 'shards') {
      this.#callbacks.onProgress?.({
        phase: 'source',
        percent: Math.min(100, progress.percent * 1.2),
        bytesLoaded: progress.bytesLoaded,
        totalBytes: progress.totalBytes,
        speed: progress.bytesPerSecond,
      });
    } else if (stage === 'layers' || stage === 'gpu_transfer') {
      this.#callbacks.onProgress?.({ phase: 'source', percent: 100, message: 'Complete' });

      const gpuPercent = 10 + (progress.percent * 0.9);
      let message;
      if (progress.layer !== undefined && progress.total) {
        message = `Layer ${progress.layer}/${progress.total}`;
      } else if (stage === 'gpu_transfer') {
        message = 'Uploading weights...';
      } else {
        message = `${Math.round(gpuPercent)}%`;
      }
      this.#callbacks.onProgress?.({ phase: 'gpu', percent: gpuPercent, message });
    } else if (stage === 'complete') {
      this.#callbacks.onProgress?.({ phase: 'source', percent: 100, message: 'Done' });
      this.#callbacks.onProgress?.({ phase: 'gpu', percent: 100, message: 'Ready' });
    }
  }

  async unload() {
    if (!this.#pipeline) return;

    log.info('ModelLoader', 'Unloading model...');

    if (typeof this.#pipeline.unload === 'function') {
      await this.#pipeline.unload();
    }

    this.#pipeline = null;
    this.#currentModel = null;
  }

  async clearAllMemory() {
    await this.unload();

    destroyBufferPool();

    const heapManager = getHeapManager();
    if (heapManager && typeof heapManager.reset === 'function') {
      heapManager.reset();
    }

    // Hint GC if available
    if (typeof globalThis.gc === 'function') {
      globalThis.gc();
    }

    log.info('ModelLoader', 'All memory cleared');
  }

  getMemoryStats() {
    if (!this.#pipeline || typeof this.#pipeline.getMemoryStats !== 'function') {
      return null;
    }
    return this.#pipeline.getMemoryStats();
  }

  getKVCacheStats() {
    if (!this.#pipeline || typeof this.#pipeline.getKVCacheStats !== 'function') {
      return null;
    }
    return this.#pipeline.getKVCacheStats();
  }

  clearKVCache() {
    if (this.#pipeline && typeof this.#pipeline.clearKVCache === 'function') {
      this.#pipeline.clearKVCache();
    }
  }
}





