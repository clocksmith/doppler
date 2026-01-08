/**
 * DOPPLER Model Manager
 * Handles model loading, unloading, initialization, and LoRA adapter management.
 */

import { getMemoryCapabilities } from '../../memory/capability.js';
import { getHeapManager } from '../../memory/heap-manager.js';
import {
  initOPFS,
  openModelDirectory,
  verifyIntegrity,
  listModels,
  loadManifestFromOPFS,
} from '../../storage/shard-manager.js';
import { getManifest, parseManifest, type RDRRManifest } from '../../storage/rdrr-format.js';
import { downloadModel } from '../../storage/downloader.js';
import { requestPersistence, getStorageReport } from '../../storage/quota.js';
import { initDevice, getKernelCapabilities, getDeviceLimits, destroyDevice, getDevice } from '../../gpu/device.js';
import { prepareKernelRuntime } from '../../gpu/kernel-runtime.js';
import { createPipeline, type InferencePipeline } from '../../inference/pipeline.js';
import { isBridgeAvailable, createBridgeClient } from '../../bridge/index.js';
import { loadLoRAFromManifest, loadLoRAFromUrl, type LoRAManifest } from '../../adapters/lora-loader.js';
import { getDopplerLoader } from '../../loader/doppler-loader.js';
import { log } from '../../debug/index.js';
import {
  DopplerCapabilities,
  type TextModelConfig,
  type InferredAttentionParams,
  type ModelEstimate,
  type LoadProgressEvent,
} from './types.js';

// Current state
let pipeline: InferencePipeline | null = null;
let currentModelId: string | null = null;

/**
 * Get the current inference pipeline
 */
export function getPipeline(): InferencePipeline | null {
  return pipeline;
}

/**
 * Get the current model ID
 */
export function getCurrentModelId(): string | null {
  return currentModelId;
}

/**
 * Extract text model configuration from manifest
 */
export function extractTextModelConfig(manifest: RDRRManifest): TextModelConfig {
  const arch = (manifest.architecture && typeof manifest.architecture === 'object')
    ? (manifest.architecture as unknown as Record<string, unknown>)
    : null;
  const cfg = manifest?.config || (manifest as unknown as Record<string, unknown>)?.modelConfig || {};
  const textCfg = (cfg as Record<string, unknown>)?.text_config || cfg;
  const textConfig = textCfg as Record<string, unknown>;

  const hiddenSize = (arch?.hiddenSize ?? textConfig.hidden_size ?? textConfig.n_embd ?? 4096) as number;

  // Try to get attention params from architecture/config, or infer from tensor shapes
  let numHeads = (arch?.numAttentionHeads ?? textConfig.num_attention_heads ?? textConfig.n_head) as number | undefined;
  let numKVHeads = (arch?.numKeyValueHeads ?? textConfig.num_key_value_heads) as number | undefined;
  let headDim = (arch?.headDim ?? textConfig.head_dim) as number | undefined;

  // If attention params missing, try to infer from tensor shapes
  if (!numHeads || !headDim) {
    const inferred = inferAttentionParams(manifest, hiddenSize);
    if (inferred) {
      numHeads = numHeads || inferred.numHeads;
      numKVHeads = numKVHeads || inferred.numKVHeads;
      headDim = headDim || inferred.headDim;
    }
  }

  // Fallback defaults
  numHeads = numHeads || 32;
  numKVHeads = numKVHeads || numHeads;
  headDim = headDim || Math.floor(hiddenSize / numHeads);

  return {
    numLayers: (arch?.numLayers ?? textConfig.num_hidden_layers ?? textConfig.n_layer ?? 32) as number,
    hiddenSize,
    intermediateSize: (arch?.intermediateSize ?? textConfig.intermediate_size ?? textConfig.n_inner ?? 14336) as number,
    numHeads,
    numKVHeads,
    headDim,
    vocabSize: (arch?.vocabSize ?? textConfig.vocab_size ?? 32000) as number,
    maxSeqLen: (arch?.maxSeqLen ?? textConfig.max_position_embeddings ?? textConfig.context_length ?? 4096) as number,
    quantization: (manifest?.quantization || 'f16').toUpperCase(),
  };
}

/**
 * Infer attention parameters from tensor shapes in manifest
 */
function inferAttentionParams(manifest: RDRRManifest, _hiddenSize: number): InferredAttentionParams | null {
  const tensors = manifest?.tensors || {};

  let qShape: number[] | null = null;
  let kShape: number[] | null = null;

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

  // Common headDim values
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

/**
 * Estimate dequantized weights size in bytes
 */
function estimateDequantizedWeightsBytes(manifest: RDRRManifest): number {
  const q = (manifest?.quantization || '').toUpperCase();
  const total = manifest?.totalSize || 0;
  if (q.startsWith('Q4')) {
    // Roughly 8x expansion when dequantized to f32.
    return total * 8;
  }
  return total;
}

// OPFS utility functions
const normalizeOPFSPath = (path: string): string => path.replace(/^\/+/, '');

const getOPFSRoot = async (): Promise<FileSystemDirectoryHandle> => {
  await initOPFS();
  if (!navigator.storage?.getDirectory) {
    throw new Error('OPFS not available');
  }
  return navigator.storage.getDirectory();
};

const resolveOPFSPath = async (
  path: string,
  createDirs: boolean
): Promise<{ dir: FileSystemDirectoryHandle; filename: string }> => {
  const normalized = normalizeOPFSPath(path);
  const parts = normalized.split('/').filter(Boolean);
  if (parts.length === 0) {
    throw new Error('Invalid OPFS path');
  }

  const filename = parts.pop() as string;
  let dir = await getOPFSRoot();

  for (const part of parts) {
    dir = await dir.getDirectoryHandle(part, { create: createDirs });
  }

  return { dir, filename };
};

export const readOPFSFile = async (path: string): Promise<ArrayBuffer> => {
  const { dir, filename } = await resolveOPFSPath(path, false);
  const handle = await dir.getFileHandle(filename);
  const file = await handle.getFile();
  return file.arrayBuffer();
};

export const writeOPFSFile = async (path: string, data: ArrayBuffer): Promise<void> => {
  const { dir, filename } = await resolveOPFSPath(path, true);
  const handle = await dir.getFileHandle(filename, { create: true });
  const writable = await handle.createWritable();
  await writable.write(data);
  await writable.close();
};

export const fetchArrayBuffer = async (url: string): Promise<ArrayBuffer> => {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: ${res.status}`);
  }
  return res.arrayBuffer();
};

/**
 * Initialize DOPPLER subsystem
 * @returns true if DOPPLER is available
 */
export async function initDoppler(): Promise<boolean> {
  if (DopplerCapabilities.initialized) {
    return DopplerCapabilities.available;
  }

  try {
    log.info('DopplerProvider', 'Initializing...');

    // Check WebGPU availability
    if (!navigator.gpu) {
      log.warn('DopplerProvider', 'WebGPU not available');
      DopplerCapabilities.initialized = true;
      return false;
    }

    // Probe memory capabilities
    const memCaps = await getMemoryCapabilities();
    DopplerCapabilities.HAS_MEMORY64 = memCaps.hasMemory64;
    DopplerCapabilities.IS_UNIFIED_MEMORY = memCaps.isUnifiedMemory;

    // Initialize WebGPU device
    const device = await initDevice();
    if (!device) {
      log.warn('DopplerProvider', 'Failed to initialize WebGPU device');
      DopplerCapabilities.initialized = true;
      return false;
    }

    // Get GPU capabilities
    const gpuCaps = getKernelCapabilities();
    DopplerCapabilities.HAS_SUBGROUPS = gpuCaps.hasSubgroups;
    DopplerCapabilities.HAS_F16 = gpuCaps.hasF16;

    // Initialize OPFS
    await initOPFS();

    // Request persistent storage
    await requestPersistence();

    // Initialize heap manager
    const heapManager = getHeapManager();
    await heapManager.init();

    // Determine tier level and max model size
    if (memCaps.isUnifiedMemory) {
      DopplerCapabilities.TIER_LEVEL = 1;
      DopplerCapabilities.TIER_NAME = 'Unified Memory';
      DopplerCapabilities.MAX_MODEL_SIZE = 60 * 1024 * 1024 * 1024; // 60GB
    } else if (memCaps.hasMemory64) {
      DopplerCapabilities.TIER_LEVEL = 2;
      DopplerCapabilities.TIER_NAME = 'Memory64';
      DopplerCapabilities.MAX_MODEL_SIZE = 40 * 1024 * 1024 * 1024; // 40GB MoE
    } else {
      DopplerCapabilities.TIER_LEVEL = 3;
      DopplerCapabilities.TIER_NAME = 'Basic';
      DopplerCapabilities.MAX_MODEL_SIZE = 8 * 1024 * 1024 * 1024; // 8GB small MoE
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

/**
 * Load a model from OPFS, download it, or access via Native Bridge
 * @param modelId - Model identifier
 * @param modelUrl - URL to download from if not cached
 * @param onProgress - Progress callback
 * @param localPath - Local file path for Native Bridge access
 */
export async function loadModel(
  modelId: string,
  modelUrl: string | null = null,
  onProgress: ((event: LoadProgressEvent) => void) | null = null,
  localPath: string | null = null
): Promise<boolean> {
  if (!DopplerCapabilities.available) {
    throw new Error('DOPPLER not initialized. Call initDoppler() first.');
  }

  try {
    log.info('DopplerProvider', `Loading model: ${modelId}`);

    let manifest: RDRRManifest | null = null;
    let useBridge = false;

    // Check if we should use Native Bridge for local path access
    if (localPath && isBridgeAvailable()) {
      log.info('DopplerProvider', `Using Native Bridge for local path: ${localPath}`);
      useBridge = true;

      try {
        const bridgeClient = await createBridgeClient();

        // Read manifest from local path
        const manifestPath = localPath.endsWith('/')
          ? `${localPath}manifest.json`
          : `${localPath}/manifest.json`;

        if (onProgress) onProgress({ stage: 'connecting', message: 'Connecting to Native Bridge...' });

        // Read manifest (use generous size - manifests are small, typically under 1MB)
        const manifestBytes = await bridgeClient.read(manifestPath, 0, 10 * 1024 * 1024);
        const manifestJson = new TextDecoder().decode(manifestBytes);
        manifest = parseManifest(manifestJson);

        log.info('DopplerProvider', `Loaded manifest via bridge: ${manifest.modelId}`);
        if (onProgress) onProgress({ stage: 'manifest', message: 'Manifest loaded via bridge' });

        // Store bridge client and local path for shard access during inference
        DopplerCapabilities.bridgeClient = bridgeClient;
        DopplerCapabilities.localPath = localPath;
      } catch (err) {
        log.error('DopplerProvider', 'Failed to load via bridge', err);
        throw new Error(`Native Bridge error: ${(err as Error).message}`);
      }
    } else {
      // Standard OPFS path
      // Open model directory
      await openModelDirectory(modelId);

      // Attempt to load manifest from OPFS (if present)
      try {
        const manifestJson = await loadManifestFromOPFS();
        manifest = parseManifest(manifestJson);
      } catch {
        manifest = null;
      }

      // Check if model exists and is valid (only if manifest loaded)
      let integrity: { valid: boolean; missingShards: number[] } = { valid: false, missingShards: [] };
      if (manifest) {
        integrity = await verifyIntegrity().catch(() => ({
          valid: false,
          missingShards: [] as number[],
        }));
      }

      if (!integrity.valid && modelUrl) {
        log.info('DopplerProvider', `Model not cached, downloading from ${modelUrl}`);
        const success = await downloadModel(modelUrl, onProgress as ((progress: unknown) => void) | undefined);
        if (!success) {
          throw new Error('Failed to download model');
        }
      } else if (!integrity.valid && !localPath) {
        throw new Error(`Model ${modelId} not found and no URL provided`);
      }

      // Get manifest
      manifest = getManifest();
    }

    if (!manifest) {
      throw new Error('Failed to load model manifest');
    }

    // Hardware/model UX estimate (approximate)
    try {
      const mc = extractTextModelConfig(manifest);
      const kvBytes = mc.numLayers * mc.maxSeqLen * mc.numKVHeads * mc.headDim * 4 * 2;
      const weightBytes = estimateDequantizedWeightsBytes(manifest);
      const estimate: ModelEstimate = {
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

    // Check model size against capabilities
    if (manifest.totalSize > DopplerCapabilities.MAX_MODEL_SIZE) {
      throw new Error(
        `Model size ${manifest.totalSize} exceeds max ${DopplerCapabilities.MAX_MODEL_SIZE}`
      );
    }

    // Check if MoE required for dGPU
    if (!DopplerCapabilities.IS_UNIFIED_MEMORY && !manifest.moeConfig) {
      log.warn('DopplerProvider', 'Dense model on discrete GPU - performance will be limited');
    }

    // Prewarm kernels once per session
    if (!DopplerCapabilities.kernelsWarmed) {
      onProgress?.({ stage: 'warming', message: 'Warming GPU kernels...' });
      await prepareKernelRuntime({ prewarm: true, prewarmMode: 'sequential' });
      DopplerCapabilities.kernelsWarmed = true;
    }

    // Kick off kernel auto-tuning in background (results cached per device)
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
        }).catch((e: Error) => {
          log.warn('DopplerProvider', 'Kernel auto-tune failed', e);
        });
      }, 0);
    }

    // Initialize pipeline with current capabilities
    const gpuCaps = getKernelCapabilities();
    const memCaps = await getMemoryCapabilities();

    // Create shard loader - use bridge or OPFS based on how model was loaded
    let loadShardFn: (idx: number) => Promise<Uint8Array>;
    if (useBridge && DopplerCapabilities.bridgeClient && DopplerCapabilities.localPath) {
      // Load shards via Native Bridge (mmap)
      const bridgeClient = DopplerCapabilities.bridgeClient;
      const basePath = DopplerCapabilities.localPath.endsWith('/')
        ? DopplerCapabilities.localPath
        : `${DopplerCapabilities.localPath}/`;

      const manifestRef = manifest; // Capture for closure
      loadShardFn = async (idx: number): Promise<Uint8Array> => {
        const shardInfo = manifestRef.shards[idx];
        if (!shardInfo) throw new Error(`Invalid shard index: ${idx}`);
        const shardPath = `${basePath}${shardInfo.filename}`;
        log.info('DopplerProvider', `Loading shard ${idx} via bridge: ${shardPath}`);
        const data = await bridgeClient.read(shardPath, 0, shardInfo.size);
        return data;
      };
    } else {
      // Load shards from OPFS
      loadShardFn = async (idx: number): Promise<Uint8Array> => {
        const m = await import('../../storage/shard-manager.js');
        const arrayBuffer = await m.loadShard(idx);
        return new Uint8Array(arrayBuffer);
      };
    }

    // Determine base URL for loading assets (tokenizer.json, etc.)
    let baseUrl: string | null = null;
    if (useBridge && DopplerCapabilities.localPath) {
      // Native Bridge: construct file:// URL or leave null for relative path handling
      baseUrl = DopplerCapabilities.localPath;
    } else if (modelUrl) {
      // Remote model: use the model URL as base
      baseUrl = modelUrl;
    }
    // For OPFS, baseUrl stays null - tokenizer.json would be fetched from same origin

    pipeline = await createPipeline(manifest as unknown as import('../../inference/pipeline/config.js').Manifest, {
      gpu: {
        capabilities: gpuCaps,
        device: getDevice(), // Use existing device, don't re-init
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

/**
 * Unload current model
 */
export async function unloadModel(): Promise<void> {
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

/**
 * Load a LoRA adapter (manifest object or URL)
 */
export async function loadLoRAAdapter(adapter: LoRAManifest | RDRRManifest | string): Promise<void> {
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
  } else if ((adapter as RDRRManifest).adapterType === 'lora' || (adapter as RDRRManifest).modelType === 'lora') {
    const loader = pipeline.dopplerLoader || getDopplerLoader();
    await loader.init();
    lora = await loader.loadLoRAWeights(adapter as RDRRManifest);
  } else {
    lora = await loadLoRAFromManifest(adapter as LoRAManifest, options);
  }

  pipeline.setLoRAAdapter(lora);
  log.info('DopplerProvider', `LoRA adapter loaded: ${lora.name}`);
}

/**
 * Unload active LoRA adapter
 */
export async function unloadLoRAAdapter(): Promise<void> {
  if (!pipeline) return;
  pipeline.setLoRAAdapter(null);
  log.info('DopplerProvider', 'LoRA adapter unloaded');
}

/**
 * Get active LoRA adapter name
 */
export function getActiveLoRA(): string | null {
  const active = pipeline?.getActiveLoRA() || null;
  return active ? active.name : null;
}

/**
 * Get list of available models
 */
export async function getAvailableModels(): Promise<string[]> {
  return listModels();
}

/**
 * Get storage info
 */
export async function getDopplerStorageInfo(): Promise<unknown> {
  // Provide quota + OPFS report
  return getStorageReport();
}

/**
 * Cleanup DOPPLER resources
 */
export async function destroyDoppler(): Promise<void> {
  await unloadModel();
  destroyDevice();

  // Disconnect bridge client if connected
  if (DopplerCapabilities.bridgeClient) {
    DopplerCapabilities.bridgeClient.disconnect();
    DopplerCapabilities.bridgeClient = null;
    DopplerCapabilities.localPath = null;
  }

  DopplerCapabilities.initialized = false;
  DopplerCapabilities.available = false;
  log.info('DopplerProvider', 'Destroyed');
}
