import { getMemoryCapabilities } from '../../memory/capability.js';
import { getHeapManager } from '../../memory/heap-manager.js';
import {
  initOPFS,
  openModelDirectory,
  verifyIntegrity,
  listModels,
  loadManifestFromOPFS,
} from '../../storage/shard-manager.js';
import { getManifest, parseManifest } from '../../storage/rdrr-format.js';
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

let pipeline = null;
let currentModelId = null;

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
  const cfg = manifest?.config || manifest?.modelConfig || {};
  const textCfg = cfg?.text_config || cfg;
  const textConfig = textCfg;

  const hiddenSize = (arch?.hiddenSize ?? textConfig.hidden_size ?? textConfig.n_embd ?? 4096);

  let numHeads = (arch?.numAttentionHeads ?? textConfig.num_attention_heads ?? textConfig.n_head);
  let numKVHeads = (arch?.numKeyValueHeads ?? textConfig.num_key_value_heads);
  let headDim = (arch?.headDim ?? textConfig.head_dim);

  if (!numHeads || !headDim) {
    const inferred = inferAttentionParams(manifest, hiddenSize);
    if (inferred) {
      numHeads = numHeads || inferred.numHeads;
      numKVHeads = numKVHeads || inferred.numKVHeads;
      headDim = headDim || inferred.headDim;
    }
  }

  numHeads = numHeads || 32;
  numKVHeads = numKVHeads || numHeads;
  headDim = headDim || Math.floor(hiddenSize / numHeads);

  return {
    numLayers: (arch?.numLayers ?? textConfig.num_hidden_layers ?? textConfig.n_layer ?? 32),
    hiddenSize,
    intermediateSize: (arch?.intermediateSize ?? textConfig.intermediate_size ?? textConfig.n_inner ?? 14336),
    numHeads,
    numKVHeads,
    headDim,
    vocabSize: (arch?.vocabSize ?? textConfig.vocab_size ?? 32000),
    maxSeqLen: (arch?.maxSeqLen ?? textConfig.max_position_embeddings ?? textConfig.context_length ?? 4096),
    quantization: (manifest?.quantization || 'f16').toUpperCase(),
  };
}

function inferAttentionParams(manifest, _hiddenSize) {
  const tensors = manifest?.tensors || {};

  let qShape = null;
  let kShape = null;

  for (const [name, tensor] of Object.entries(tensors)) {
    if (name.includes('layers.0.self_attn.q_proj.weight') || name.includes('layers.0.attention.q_proj.weight')) {
      qShape = tensor.shape;
    }
    if (name.includes('layers.0.self_attn.k_proj.weight') || name.includes('layers.0.attention.k_proj.weight')) {
      kShape = tensor.shape;
    }
    if (qShape && kShape) break;
  }

  if (!qShape || !kShape) return null;

  const qOutDim = qShape[0];
  const kOutDim = kShape[0];

  const commonHeadDims = [256, 128, 160, 64, 96, 80];

  for (const testHeadDim of commonHeadDims) {
    if (qOutDim % testHeadDim === 0 && kOutDim % testHeadDim === 0) {
      const numHeads = qOutDim / testHeadDim;
      const numKVHeads = kOutDim / testHeadDim;
      if (numHeads >= numKVHeads && numHeads > 0 && numKVHeads > 0) {
        return { numHeads, numKVHeads, headDim: testHeadDim };
      }
    }
  }

  return null;
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
  await initOPFS();
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

    await initOPFS();
    await requestPersistence();

    const heapManager = getHeapManager();
    await heapManager.init();

    if (memCaps.isUnifiedMemory) {
      DopplerCapabilities.TIER_LEVEL = 1;
      DopplerCapabilities.TIER_NAME = 'Unified Memory';
      DopplerCapabilities.MAX_MODEL_SIZE = 60 * 1024 * 1024 * 1024;
    } else if (memCaps.hasMemory64) {
      DopplerCapabilities.TIER_LEVEL = 2;
      DopplerCapabilities.TIER_NAME = 'Memory64';
      DopplerCapabilities.MAX_MODEL_SIZE = 40 * 1024 * 1024 * 1024;
    } else {
      DopplerCapabilities.TIER_LEVEL = 3;
      DopplerCapabilities.TIER_NAME = 'Basic';
      DopplerCapabilities.MAX_MODEL_SIZE = 8 * 1024 * 1024 * 1024;
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

        const manifestBytes = await bridgeClient.read(manifestPath, 0, 10 * 1024 * 1024);
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
      await openModelDirectory(modelId);

      try {
        const manifestJson = await loadManifestFromOPFS();
        manifest = parseManifest(manifestJson);
      } catch {
        manifest = null;
      }

      let integrity = { valid: false, missingShards: [] };
      if (manifest) {
        integrity = await verifyIntegrity().catch(() => ({
          valid: false,
          missingShards: [],
        }));
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
