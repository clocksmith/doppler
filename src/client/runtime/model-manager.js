import { getMemoryCapabilities } from '../../memory/capability.js';
import { getHeapManager } from '../../memory/heap-manager.js';
import {
  initStorage,
  openModelStore,
  verifyIntegrity,
  listModels,
  loadManifestFromStore,
} from '../../storage/shard-manager.js';
import { getManifest, parseManifest, getManifestUrl } from '../../formats/rdrr/index.js';
import { downloadModel } from '../../storage/downloader.js';
import { requestPersistence, getStorageReport } from '../../storage/quota.js';
import { initDevice, getKernelCapabilities, getDeviceLimits, destroyDevice, getDevice } from '../../gpu/device.js';
import { prepareKernelRuntime } from '../../gpu/kernel-runtime.js';
import { createPipeline } from '../../inference/pipelines/text.js';
import { getDopplerLoader } from '../../loader/doppler-loader.js';
import { log } from '../../debug/index.js';
import { DopplerCapabilities } from './types.js';
import { GB, HEADER_READ_SIZE } from '../../config/schema/index.js';
import { resolveBridgeSourceRuntimeBundle } from './source-runtime.js';
import { getRuntimeConfig } from '../../config/runtime.js';
import {
  buildSourceArtifactFingerprint,
  createStoredSourceArtifactContext,
  synthesizeStoredSourceArtifactManifest,
  verifyStoredSourceArtifact,
} from '../../storage/source-artifact-store.js';

let pipeline = null;
let currentModelId = null;
let bridgeModulePromise = null;
let loraModulePromise = null;

async function getExperimentalBridgeModule() {
  bridgeModulePromise ??= import('../../experimental/bridge/index.js');
  return bridgeModulePromise;
}

async function getExperimentalLoRAModule() {
  loraModulePromise ??= import('../../experimental/adapters/lora-loader.js');
  return loraModulePromise;
}

function manifestsDiffer(localManifest, remoteManifest) {
  if (!localManifest || !remoteManifest) return true;
  if (localManifest.modelId !== remoteManifest.modelId) return true;
  if (localManifest.quantization !== remoteManifest.quantization) return true;
  if (localManifest.hashAlgorithm !== remoteManifest.hashAlgorithm) return true;
  if (localManifest.totalSize !== remoteManifest.totalSize) return true;

  const localShards = Array.isArray(localManifest.shards) ? localManifest.shards : [];
  const remoteShards = Array.isArray(remoteManifest.shards) ? remoteManifest.shards : [];
  if (localShards.length !== remoteShards.length) return true;
  if (buildSourceArtifactFingerprint(localManifest) !== buildSourceArtifactFingerprint(remoteManifest)) {
    return true;
  }

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

export async function verifyExplicitModelUrlMatch(
  localManifest,
  modelUrl,
  fetchRemoteManifest = tryFetchRemoteManifest
) {
  if (!localManifest || !modelUrl) {
    return;
  }
  let remoteManifest = null;
  try {
    remoteManifest = await fetchRemoteManifest(modelUrl);
  } catch (error) {
    throw new Error(
      `Could not compare cached manifest with explicit modelUrl "${modelUrl}": ${error.message}`
    );
  }
  if (remoteManifest && manifestsDiffer(localManifest, remoteManifest)) {
    throw new Error(
      `Explicit modelUrl "${modelUrl}" does not match the cached manifest for "${localManifest.modelId ?? 'unknown'}". ` +
      'Clear the cache or load the matching source explicitly.'
    );
  }
}

export function shouldAutoTuneKernels(runtimeConfig = getRuntimeConfig()) {
  return runtimeConfig?.shared?.kernelWarmup?.autoTune === true;
}

export function getPipeline() {
  return pipeline;
}

export function getCurrentModelId() {
  return currentModelId;
}

function requireManifestQuantization(manifest) {
  const quantization = String(manifest?.quantization ?? '').trim();
  if (!quantization) {
    throw new Error('Manifest is missing quantization; re-convert the model.');
  }
  return quantization.toUpperCase();
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
    quantization: requireManifestQuantization(manifest),
  };
}

function estimateDequantizedWeightsBytes(manifest) {
  const q = requireManifestQuantization(manifest);
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

async function initDoppler() {
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

    let runtimeModel = null;
    let useBridge = false;
    let bridgeStorageContext = null;
    let bridgeSourceMode = false;

    if (localPath) {
      const { isBridgeAvailable, createBridgeClient } = await getExperimentalBridgeModule();
      if (!isBridgeAvailable()) {
        throw new Error('Local path loading requires the experimental bridge surface, but no bridge is available.');
      }
      log.info('DopplerProvider', `Using Native Bridge for local path: ${localPath}`);
      useBridge = true;

      try {
        const bridgeClient = await createBridgeClient();

        const manifestPath = localPath.endsWith('/')
          ? `${localPath}manifest.json`
          : `${localPath}/manifest.json`;

        if (onProgress) onProgress({ stage: 'connecting', message: 'Connecting to Native Bridge...' });

        try {
          const manifestBytes = await bridgeClient.read(manifestPath, 0, HEADER_READ_SIZE);
          const manifestJson = new TextDecoder().decode(manifestBytes);
          runtimeModel = parseManifest(manifestJson);
          log.info('DopplerProvider', `Loaded manifest via bridge: ${runtimeModel.modelId}`);
          if (onProgress) onProgress({ stage: 'manifest', message: 'Manifest loaded via bridge' });
          const persistedSourceBundle = await resolveBridgeSourceRuntimeBundle({
            bridgeClient,
            localPath,
            modelId,
            model: runtimeModel,
            verifyHashes: true,
            onProgress: (progress) => onProgress?.(progress),
          });
          if (persistedSourceBundle) {
            bridgeStorageContext = persistedSourceBundle.storageContext;
            bridgeSourceMode = true;
            if (onProgress) {
              onProgress({
                stage: 'manifest',
                message: `Direct-source runtime model ready (${persistedSourceBundle.sourceKind} artifact mode)`,
              });
            }
          }
        } catch (manifestError) {
          log.warn(
            'DopplerProvider',
            `Bridge manifest probe failed, trying direct source runtime: ${manifestError.message}`
          );
          const sourceBundle = await resolveBridgeSourceRuntimeBundle({
            bridgeClient,
            localPath,
            modelId,
            verifyHashes: true,
            onProgress: (progress) => onProgress?.(progress),
          });
          if (!sourceBundle) {
            throw manifestError;
          }
          runtimeModel = sourceBundle.model ?? sourceBundle.manifest;
          bridgeStorageContext = sourceBundle.storageContext;
          bridgeSourceMode = true;
          if (onProgress) {
            onProgress({
              stage: 'manifest',
              message: `Runtime model ready (${sourceBundle.sourceKind} direct-source mode)`,
            });
          }
        }

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
        runtimeModel = parseManifest(manifestJson);
      } catch {
        runtimeModel = null;
      }

      let integrity = { valid: false, missingShards: [] };
      if (runtimeModel) {
        const sourceArtifactFingerprint = buildSourceArtifactFingerprint(runtimeModel);
        if (sourceArtifactFingerprint) {
          const sourceIntegrity = await verifyStoredSourceArtifact(runtimeModel, { checkHashes: false }).catch(() => ({
            valid: false,
            missingFiles: [],
          }));
          integrity = {
            valid: sourceIntegrity.valid,
            missingShards: Array.isArray(sourceIntegrity.missingFiles) ? sourceIntegrity.missingFiles : [],
          };
        } else {
          integrity = await verifyIntegrity({ checkHashes: false }).catch(() => ({
            valid: false,
            missingShards: [],
          }));
        }
      }

      if (integrity.valid && runtimeModel && modelUrl) {
        await verifyExplicitModelUrlMatch(runtimeModel, modelUrl);
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

      runtimeModel = getManifest();
    }

    if (!runtimeModel) {
      throw new Error('Failed to load model manifest');
    }

    const synthesizedRuntimeModel = synthesizeStoredSourceArtifactManifest(runtimeModel);
    if (synthesizedRuntimeModel.changed) {
      log.info(
        'DopplerProvider',
        `Enabled stored-shard source runtime for "${runtimeModel.modelId ?? modelId}" warm loads`
      );
      runtimeModel = synthesizedRuntimeModel.manifest;
    }

    try {
      const mc = extractTextModelConfig(runtimeModel);
      const kvBytes = mc.numLayers * mc.maxSeqLen * mc.numKVHeads * mc.headDim * 4 * 2;
      const weightBytes = estimateDequantizedWeightsBytes(runtimeModel);
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

    if (runtimeModel.totalSize > DopplerCapabilities.MAX_MODEL_SIZE) {
      throw new Error(
        `Model size ${runtimeModel.totalSize} exceeds max ${DopplerCapabilities.MAX_MODEL_SIZE}`
      );
    }

    if (!DopplerCapabilities.IS_UNIFIED_MEMORY && !runtimeModel.moeConfig) {
      log.warn('DopplerProvider', 'Dense model on discrete GPU - performance will be limited');
    }

    if (!DopplerCapabilities.kernelsWarmed) {
      onProgress?.({ stage: 'warming', message: 'Warming GPU kernels...' });
      await prepareKernelRuntime({ prewarm: true, prewarmMode: 'sequential' });
      DopplerCapabilities.kernelsWarmed = true;
    }

    if (
      !DopplerCapabilities.kernelsTuned
      && shouldAutoTuneKernels()
      && typeof setTimeout !== 'undefined'
    ) {
      DopplerCapabilities.kernelsTuned = true;
      const tuneConfig = extractTextModelConfig(runtimeModel);
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

    let storageContext = bridgeStorageContext;
    if (!storageContext && buildSourceArtifactFingerprint(runtimeModel)) {
      storageContext = createStoredSourceArtifactContext(runtimeModel, { verifyHashes: true });
    }
    if (!storageContext && useBridge && DopplerCapabilities.bridgeClient && DopplerCapabilities.localPath) {
      const bridgeClient = DopplerCapabilities.bridgeClient;
      const basePath = DopplerCapabilities.localPath.endsWith('/')
        ? DopplerCapabilities.localPath
        : `${DopplerCapabilities.localPath}/`;

      const runtimeModelRef = runtimeModel;
      const resolveShard = (idx) => {
        const shardInfo = runtimeModelRef.shards[idx];
        if (!shardInfo) throw new Error(`Invalid shard index: ${idx}`);
        return {
          shardInfo,
          shardPath: `${basePath}${shardInfo.filename}`,
        };
      };

      const loadShard = async (idx) => {
        const { shardInfo, shardPath } = resolveShard(idx);
        log.info('DopplerProvider', `Loading shard ${idx} via bridge: ${shardPath}`);
        return bridgeClient.read(shardPath, 0, shardInfo.size);
      };

      const loadShardRange = async (idx, offset, length = null) => {
        const { shardInfo, shardPath } = resolveShard(idx);
        const startRaw = Number(offset);
        const start = Number.isFinite(startRaw) ? Math.max(0, Math.floor(startRaw)) : 0;
        const maxLength = Math.max(0, shardInfo.size - start);
        const requested = length == null
          ? maxLength
          : Math.max(0, Math.min(Math.floor(Number(length) || 0), maxLength));
        if (requested <= 0) {
          return new ArrayBuffer(0);
        }
        return bridgeClient.read(shardPath, start, requested);
      };

      const streamShardRange = async function* (idx, offset = 0, length = null, options = {}) {
        const chunkRaw = Number(options?.chunkBytes);
        const chunkBytes = Number.isFinite(chunkRaw) && chunkRaw > 0
          ? Math.floor(chunkRaw)
          : 4 * 1024 * 1024;
        let produced = 0;
        while (true) {
          const remaining = length == null ? chunkBytes : Math.max(0, length - produced);
          if (length != null && remaining <= 0) break;
          const requestLength = Math.min(chunkBytes, remaining);
          const chunk = await loadShardRange(idx, offset + produced, requestLength);
          const bytes = chunk instanceof Uint8Array ? chunk : new Uint8Array(chunk);
          if (bytes.byteLength === 0) break;
          produced += bytes.byteLength;
          yield bytes;
          if (bytes.byteLength < requestLength) break;
        }
      };

      storageContext = {
        loadShard,
        loadShardRange,
        streamShardRange,
        verifyHashes: false,
      };
    }

    let baseUrl = null;
    if (useBridge && DopplerCapabilities.localPath && !bridgeSourceMode) {
      baseUrl = DopplerCapabilities.localPath;
    } else if (modelUrl) {
      baseUrl = modelUrl;
    }

    const pipelineContexts = {
      gpu: {
        capabilities: gpuCaps,
        device: getDevice(),
      },
      memory: {
        capabilities: memCaps,
        heapManager: getHeapManager(),
      },
      baseUrl,
    };
    if (storageContext) {
      pipelineContexts.storage = storageContext;
    }

    pipeline = await createPipeline(runtimeModel, pipelineContexts);

    currentModelId = modelId;
    DopplerCapabilities.currentModelId = modelId;
    log.info('DopplerProvider', `Model loaded: ${modelId}`);
    return true;
  } catch (err) {
    log.error('DopplerProvider', 'Failed to load model', err);
    throw err;
  }
}

async function loadLoRAAdapter(adapter) {
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
    const { loadLoRAFromUrl } = await getExperimentalLoRAModule();
    lora = await loadLoRAFromUrl(adapter, options);
  } else if (adapter.adapterType === 'lora' || adapter.modelType === 'lora') {
    const loader = pipeline.dopplerLoader || getDopplerLoader();
    await loader.init();
    lora = await loader.loadLoRAWeights(adapter);
  } else {
    const { loadLoRAFromManifest } = await getExperimentalLoRAModule();
    lora = await loadLoRAFromManifest(adapter, options);
  }

  pipeline.setLoRAAdapter(lora);
  log.info('DopplerProvider', `LoRA adapter loaded: ${lora.name}`);
}

export async function activateLoRAFromTrainingOutput(trainingOutput) {
  if (!pipeline) {
    return {
      activated: false,
      adapterName: null,
      source: null,
      reason: 'no_model_loaded',
    };
  }

  const output = trainingOutput && typeof trainingOutput === 'object'
    ? trainingOutput
    : null;
  if (!output && typeof trainingOutput !== 'string') {
    return {
      activated: false,
      adapterName: getActiveLoRA(),
      source: null,
      reason: 'no_adapter_candidate',
    };
  }

  if (typeof trainingOutput === 'string') {
    await loadLoRAAdapter(trainingOutput);
    return {
      activated: true,
      adapterName: getActiveLoRA(),
      source: 'adapter-string',
      reason: null,
    };
  }

  if (output.adapterManifest && typeof output.adapterManifest === 'object') {
    await loadLoRAAdapter(output.adapterManifest);
    return {
      activated: true,
      adapterName: getActiveLoRA(),
      source: 'adapterManifest',
      reason: null,
    };
  }

  if (typeof output.adapterManifestJson === 'string' && output.adapterManifestJson.trim()) {
    const manifest = JSON.parse(output.adapterManifestJson);
    await loadLoRAAdapter(manifest);
    return {
      activated: true,
      adapterName: getActiveLoRA(),
      source: 'adapterManifestJson',
      reason: null,
    };
  }

  if (typeof output.adapterManifestUrl === 'string' && output.adapterManifestUrl.trim()) {
    await loadLoRAAdapter(output.adapterManifestUrl.trim());
    return {
      activated: true,
      adapterName: getActiveLoRA(),
      source: 'adapterManifestUrl',
      reason: null,
    };
  }

  if (typeof output.adapterManifestPath === 'string' && output.adapterManifestPath.trim()) {
    const path = output.adapterManifestPath.trim();
    if (path.startsWith('http://') || path.startsWith('https://')) {
      await loadLoRAAdapter(path);
      return {
        activated: true,
        adapterName: getActiveLoRA(),
        source: 'adapterManifestPath:url',
        reason: null,
      };
    }
    const isNode = typeof process !== 'undefined' && !!process.versions?.node;
    if (!isNode) {
      throw new Error('adapterManifestPath local files require Node runtime.');
    }
    const { readFile } = await import('node:fs/promises');
    const raw = await readFile(path, 'utf8');
    const manifest = JSON.parse(raw);
    await loadLoRAAdapter(manifest);
    return {
      activated: true,
      adapterName: getActiveLoRA(),
      source: 'adapterManifestPath:file',
      reason: null,
    };
  }

  if (output.adapter != null) {
    await loadLoRAAdapter(output.adapter);
    return {
      activated: true,
      adapterName: getActiveLoRA(),
      source: 'adapter',
      reason: null,
    };
  }

  return {
    activated: false,
    adapterName: getActiveLoRA(),
    source: null,
    reason: 'no_adapter_candidate',
  };
}

export function getActiveLoRA() {
  const active = pipeline?.getActiveLoRA() || null;
  return active ? active.name : null;
}

