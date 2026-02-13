import { getMemoryCapabilities } from '../../memory/capability.js';
import { getHeapManager } from '../../memory/heap-manager.js';
import {
  initStorage,
  openModelStore,
  verifyIntegrity,
  listModels,
  loadManifestFromStore,
} from '../../storage/shard-manager.js';
import { getManifest, parseManifest, getManifestUrl } from '../../storage/rdrr-format.js';
import { downloadModel } from '../../storage/downloader.js';
import { requestPersistence, getStorageReport } from '../../storage/quota.js';
import { initDevice, getKernelCapabilities, getDeviceLimits, destroyDevice, getDevice } from '../../gpu/device.js';
import { prepareKernelRuntime } from '../../gpu/kernel-runtime.js';
import { createPipeline } from '../../inference/pipeline.js';
import { isBridgeAvailable, createBridgeClient } from '../../bridge/index.js';
import { loadLoRAFromManifest, loadLoRAFromUrl } from '../../adapters/lora-loader.js';
import { getDopplerLoader } from '../../loader/doppler-loader.js';
import { log } from '../../debug/index.js';
import { DopplerCapabilities } from './types.js';
import { GB, HEADER_READ_SIZE } from '../../config/schema/index.js';

let pipeline = null;
let currentModelId = null;

function manifestsDiffer(localManifest, remoteManifest) {
  if (!localManifest || !remoteManifest) return true;
  if (localManifest.modelId !== remoteManifest.modelId) return true;
  if (localManifest.quantization !== remoteManifest.quantization) return true;
  if (localManifest.hashAlgorithm !== remoteManifest.hashAlgorithm) return true;
  if (localManifest.totalSize !== remoteManifest.totalSize) return true;

  const localShards = Array.isArray(localManifest.shards) ? localManifest.shards : [];
  const remoteShards = Array.isArray(remoteManifest.shards) ? remoteManifest.shards : [];
  if (localShards.length !== remoteShards.length) return true;

  for (let i = 0; i < localShards.length; i++) {
    const local = localShards[i];
    const remote = remoteShards[i];
    if (!local || !remote) return true;
    if (local.size !== remote.size) return true;
    if (local.hash !== remote.hash) return true;
    if (local.filename !== remote.filename) return true;
  }

  return false;
}

async function tryFetchRemoteManifest(modelUrl) {
  if (!modelUrl) return null;
  const response = await fetch(getManifestUrl(modelUrl));
  if (!response.ok) {
    throw new Error(`Failed to fetch remote manifest: ${response.status}`);
  }
  const manifestJson = await response.text();
  const manifest = JSON.parse(manifestJson);
  if (!manifest || typeof manifest !== 'object' || !Array.isArray(manifest.shards)) {
    throw new Error('Remote manifest is invalid');
  }
  return manifest;
}

export function getPipeline() {
  return pipeline;
}

export function getCurrentModelId() {
  return currentModelId;
}

export function extractTextModelConfig(manifest) {
  const arch = (manifest.architecture && typeof manifest.architecture === 'object')
    ? manifest.architecture
    : null;
  if (!arch) {
    throw new Error('Manifest is missing architecture config; re-convert the model.');
  }

  return {
    numLayers: arch.numLayers,
    hiddenSize: arch.hiddenSize,
    intermediateSize: arch.intermediateSize,
    numHeads: arch.numAttentionHeads,
    numKVHeads: arch.numKeyValueHeads,
    headDim: arch.headDim,
    vocabSize: arch.vocabSize,
    maxSeqLen: arch.maxSeqLen,
    quantization: (manifest?.quantization || 'f16').toUpperCase(),
  };
}

function estimateDequantizedWeightsBytes(manifest) {
  const q = (manifest?.quantization || '').toUpperCase();
  const total = manifest?.totalSize || 0;
  if (q.startsWith('Q4')) {
    return total * 8;
  }
  return total;
}

const normalizeOPFSPath = (path) => path.replace(/^\/+/, '');

const getOPFSRoot = async () => {
  await initStorage();
  if (!navigator.storage?.getDirectory) {
    throw new Error('OPFS not available');
  }
  return navigator.storage.getDirectory();
};

const resolveOPFSPath = async (path, createDirs) => {
  const normalized = normalizeOPFSPath(path);
  const parts = normalized.split('/').filter(Boolean);
  if (parts.length === 0) {
    throw new Error('Invalid OPFS path');
  }

  const filename = parts.pop();
  let dir = await getOPFSRoot();

  for (const part of parts) {
    dir = await dir.getDirectoryHandle(part, { create: createDirs });
  }

  return { dir, filename };
};

export const readOPFSFile = async (path) => {
  const { dir, filename } = await resolveOPFSPath(path, false);
  const handle = await dir.getFileHandle(filename);
  const file = await handle.getFile();
  return file.arrayBuffer();
};

export const writeOPFSFile = async (path, data) => {
  const { dir, filename } = await resolveOPFSPath(path, true);
  const handle = await dir.getFileHandle(filename, { create: true });
  const writable = await handle.createWritable();
  await writable.write(data);
  await writable.close();
};

export const fetchArrayBuffer = async (url) => {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: ${res.status}`);
  }
  return res.arrayBuffer();
};

export async function initDoppler() {
  if (DopplerCapabilities.initialized) {
    return DopplerCapabilities.available;
  }

  try {
    log.info('DopplerProvider', 'Initializing...');

    if (!navigator.gpu) {
      log.warn('DopplerProvider', 'WebGPU not available');
      DopplerCapabilities.initialized = true;
      return false;
    }

    const memCaps = await getMemoryCapabilities();
    DopplerCapabilities.HAS_MEMORY64 = memCaps.hasMemory64;
    DopplerCapabilities.IS_UNIFIED_MEMORY = memCaps.isUnifiedMemory;

    const device = await initDevice();
    if (!device) {
      log.warn('DopplerProvider', 'Failed to initialize WebGPU device');
      DopplerCapabilities.initialized = true;
      return false;
    }

    const gpuCaps = getKernelCapabilities();
    DopplerCapabilities.HAS_SUBGROUPS = gpuCaps.hasSubgroups;
    DopplerCapabilities.HAS_F16 = gpuCaps.hasF16;

    await initStorage();
    await requestPersistence();

    const heapManager = getHeapManager();
    await heapManager.init();

    if (memCaps.isUnifiedMemory) {
      DopplerCapabilities.TIER_LEVEL = 1;
      DopplerCapabilities.TIER_NAME = 'Unified Memory';
      DopplerCapabilities.MAX_MODEL_SIZE = 60 * GB;
    } else if (memCaps.hasMemory64) {
      DopplerCapabilities.TIER_LEVEL = 2;
      DopplerCapabilities.TIER_NAME = 'Memory64';
      DopplerCapabilities.MAX_MODEL_SIZE = 40 * GB;
    } else {
      DopplerCapabilities.TIER_LEVEL = 3;
      DopplerCapabilities.TIER_NAME = 'Basic';
      DopplerCapabilities.MAX_MODEL_SIZE = 8 * GB;
    }

    DopplerCapabilities.available = true;
    DopplerCapabilities.initialized = true;

    log.info('DopplerProvider', 'Initialized successfully', DopplerCapabilities);
    return true;
  } catch (err) {
    log.error('DopplerProvider', 'Init failed', err);
    DopplerCapabilities.initialized = true;
    DopplerCapabilities.available = false;
    return false;
  }
}

export async function loadModel(modelId, modelUrl = null, onProgress = null, localPath = null) {
  if (!DopplerCapabilities.available) {
    throw new Error('DOPPLER not initialized. Call initDoppler() first.');
  }

  try {
    log.info('DopplerProvider', `Loading model: ${modelId}`);

    let manifest = null;
    let useBridge = false;

    if (localPath && isBridgeAvailable()) {
      log.info('DopplerProvider', `Using Native Bridge for local path: ${localPath}`);
      useBridge = true;

      try {
        const bridgeClient = await createBridgeClient();

        const manifestPath = localPath.endsWith('/')
          ? `${localPath}manifest.json`
          : `${localPath}/manifest.json`;

        if (onProgress) onProgress({ stage: 'connecting', message: 'Connecting to Native Bridge...' });

        const manifestBytes = await bridgeClient.read(manifestPath, 0, HEADER_READ_SIZE);
        const manifestJson = new TextDecoder().decode(manifestBytes);
        manifest = parseManifest(manifestJson);

        log.info('DopplerProvider', `Loaded manifest via bridge: ${manifest.modelId}`);
        if (onProgress) onProgress({ stage: 'manifest', message: 'Manifest loaded via bridge' });

        DopplerCapabilities.bridgeClient = bridgeClient;
        DopplerCapabilities.localPath = localPath;
      } catch (err) {
        log.error('DopplerProvider', 'Failed to load via bridge', err);
        throw new Error(`Native Bridge error: ${err.message}`);
      }
    } else {
      await openModelStore(modelId);

      try {
        const manifestJson = await loadManifestFromStore();
        manifest = parseManifest(manifestJson);
      } catch {
        manifest = null;
      }

      let integrity = { valid: false, missingShards: [] };
      if (manifest) {
        integrity = await verifyIntegrity({ checkHashes: false }).catch(() => ({
          valid: false,
          missingShards: [],
        }));
      }

      if (integrity.valid && manifest && modelUrl) {
        try {
          const remoteManifest = await tryFetchRemoteManifest(modelUrl);
          if (remoteManifest && manifestsDiffer(manifest, remoteManifest)) {
            log.info('DopplerProvider', 'Cached model differs from source URL manifest; refreshing cache');
            integrity = { valid: false, missingShards: [] };
          }
        } catch (error) {
          log.warn(
            'DopplerProvider',
            `Could not compare cached manifest with source URL (${error.message}); using cached model`
          );
        }
      }

      if (!integrity.valid && modelUrl) {
        log.info('DopplerProvider', `Model not cached, downloading from ${modelUrl}`);
        const success = await downloadModel(modelUrl, onProgress);
        if (!success) {
          throw new Error('Failed to download model');
        }
      } else if (!integrity.valid && !localPath) {
        throw new Error(`Model ${modelId} not found and no URL provided`);
      }

      manifest = getManifest();
    }

    if (!manifest) {
      throw new Error('Failed to load model manifest');
    }

    try {
      const mc = extractTextModelConfig(manifest);
      const kvBytes = mc.numLayers * mc.maxSeqLen * mc.numKVHeads * mc.headDim * 4 * 2;
      const weightBytes = estimateDequantizedWeightsBytes(manifest);
      const estimate = {
        weightsBytes: weightBytes,
        kvCacheBytes: kvBytes,
        totalBytes: weightBytes + kvBytes,
        modelConfig: mc,
      };
      DopplerCapabilities.lastModelEstimate = estimate;

      const limits = getDeviceLimits();
      if (limits?.maxBufferSize && estimate.totalBytes > limits.maxBufferSize * 0.8) {
        log.warn('DopplerProvider', 'Estimated GPU usage near device limits');
      }
      onProgress?.({
        stage: 'estimate',
        message: 'Estimated GPU memory usage computed',
        estimate,
      });
    } catch (e) {
      log.warn('DopplerProvider', 'Failed to estimate GPU memory', e);
    }

    if (manifest.totalSize > DopplerCapabilities.MAX_MODEL_SIZE) {
      throw new Error(
        `Model size ${manifest.totalSize} exceeds max ${DopplerCapabilities.MAX_MODEL_SIZE}`
      );
    }

    if (!DopplerCapabilities.IS_UNIFIED_MEMORY && !manifest.moeConfig) {
      log.warn('DopplerProvider', 'Dense model on discrete GPU - performance will be limited');
    }

    if (!DopplerCapabilities.kernelsWarmed) {
      onProgress?.({ stage: 'warming', message: 'Warming GPU kernels...' });
      await prepareKernelRuntime({ prewarm: true, prewarmMode: 'sequential' });
      DopplerCapabilities.kernelsWarmed = true;
    }

    if (!DopplerCapabilities.kernelsTuned && typeof setTimeout !== 'undefined') {
      DopplerCapabilities.kernelsTuned = true;
      const tuneConfig = extractTextModelConfig(manifest);
      setTimeout(() => {
        prepareKernelRuntime({
          prewarm: false,
          autoTune: true,
          modelConfig: {
            hiddenSize: tuneConfig.hiddenSize,
            intermediateSize: tuneConfig.intermediateSize,
            numHeads: tuneConfig.numHeads,
            numKVHeads: tuneConfig.numKVHeads,
            headDim: tuneConfig.headDim,
          },
        }).catch((e) => {
          log.warn('DopplerProvider', 'Kernel auto-tune failed', e);
        });
      }, 0);
    }

    const gpuCaps = getKernelCapabilities();
    const memCaps = await getMemoryCapabilities();

    let loadShardFn;
    if (useBridge && DopplerCapabilities.bridgeClient && DopplerCapabilities.localPath) {
      const bridgeClient = DopplerCapabilities.bridgeClient;
      const basePath = DopplerCapabilities.localPath.endsWith('/')
        ? DopplerCapabilities.localPath
        : `${DopplerCapabilities.localPath}/`;

      const manifestRef = manifest;
      loadShardFn = async (idx) => {
        const shardInfo = manifestRef.shards[idx];
        if (!shardInfo) throw new Error(`Invalid shard index: ${idx}`);
        const shardPath = `${basePath}${shardInfo.filename}`;
        log.info('DopplerProvider', `Loading shard ${idx} via bridge: ${shardPath}`);
        const data = await bridgeClient.read(shardPath, 0, shardInfo.size);
        return data;
      };
    } else {
      loadShardFn = async (idx) => {
        const m = await import('../../storage/shard-manager.js');
        const arrayBuffer = await m.loadShard(idx);
        return new Uint8Array(arrayBuffer);
      };
    }

    let baseUrl = null;
    if (useBridge && DopplerCapabilities.localPath) {
      baseUrl = DopplerCapabilities.localPath;
    } else if (modelUrl) {
      baseUrl = modelUrl;
    }

    pipeline = await createPipeline(manifest, {
      gpu: {
        capabilities: gpuCaps,
        device: getDevice(),
      },
      memory: {
        capabilities: memCaps,
        heapManager: getHeapManager(),
      },
      storage: {
        loadShard: loadShardFn,
      },
      baseUrl,
    });

    currentModelId = modelId;
    DopplerCapabilities.currentModelId = modelId;
    log.info('DopplerProvider', `Model loaded: ${modelId}`);
    return true;
  } catch (err) {
    log.error('DopplerProvider', 'Failed to load model', err);
    throw err;
  }
}

export async function unloadModel() {
  if (pipeline) {
    if (typeof pipeline.unload === 'function') {
      await pipeline.unload();
    }
    pipeline = null;
  }
  currentModelId = null;
  DopplerCapabilities.currentModelId = null;
  log.info('DopplerProvider', 'Model unloaded');
}

export async function loadLoRAAdapter(adapter) {
  if (!pipeline) {
    throw new Error('No model loaded. Call loadModel() first.');
  }

  const options = {
    readOPFS: readOPFSFile,
    writeOPFS: writeOPFSFile,
    fetchUrl: fetchArrayBuffer,
  };

  let lora;
  if (typeof adapter === 'string') {
    lora = await loadLoRAFromUrl(adapter, options);
  } else if (adapter.adapterType === 'lora' || adapter.modelType === 'lora') {
    const loader = pipeline.dopplerLoader || getDopplerLoader();
    await loader.init();
    lora = await loader.loadLoRAWeights(adapter);
  } else {
    lora = await loadLoRAFromManifest(adapter, options);
  }

  pipeline.setLoRAAdapter(lora);
  log.info('DopplerProvider', `LoRA adapter loaded: ${lora.name}`);
}

export async function unloadLoRAAdapter() {
  if (!pipeline) return;
  pipeline.setLoRAAdapter(null);
  log.info('DopplerProvider', 'LoRA adapter unloaded');
}

export function getActiveLoRA() {
  const active = pipeline?.getActiveLoRA() || null;
  return active ? active.name : null;
}

export async function getAvailableModels() {
  return listModels();
}

export async function getDopplerStorageInfo() {
  return getStorageReport();
}

export async function destroyDoppler() {
  await unloadModel();
  destroyDevice();

  if (DopplerCapabilities.bridgeClient) {
    DopplerCapabilities.bridgeClient.disconnect();
    DopplerCapabilities.bridgeClient = null;
    DopplerCapabilities.localPath = null;
  }

  DopplerCapabilities.initialized = false;
  DopplerCapabilities.available = false;
  log.info('DopplerProvider', 'Destroyed');
}
