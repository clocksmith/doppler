/**
 * DOPPLER Provider - LLM Client Integration
 * Registers DOPPLER as a local WebGPU option in llm-client.js
 */

import { getMemoryCapabilities, type MemoryCapabilities } from '../src/memory/capability.js';
import { getHeapManager } from '../src/memory/heap-manager.js';
import {
  initOPFS,
  openModelDirectory,
  verifyIntegrity,
  listModels,
  loadManifestFromOPFS,
} from '../src/storage/shard-manager.js';
import { getManifest, parseManifest, type RDRRManifest } from '../src/storage/rdrr-format.js';
import { downloadModel } from '../src/storage/downloader.js';
import { requestPersistence, getStorageReport } from '../src/storage/quota.js';
import { initDevice, getKernelCapabilities, getDeviceLimits, destroyDevice } from '../src/gpu/device.js';
import { prepareKernelRuntime } from '../src/gpu/kernel-runtime.js';
import { createPipeline, type InferencePipeline, type KVCacheSnapshot } from '../src/inference/pipeline.js';
import { isBridgeAvailable, createBridgeClient, type ExtensionBridgeClient } from '../src/bridge/index.js';
import { loadLoRAFromManifest, loadLoRAFromUrl, type LoRAManifest } from '../src/adapters/lora-loader.js';
import { getDopplerLoader } from '../src/loader/doppler-loader.js';
import { log } from '../src/debug/index.js';

export const DOPPLER_PROVIDER_VERSION = '0.1.0';

/**
 * Text model configuration extracted from manifest
 */
export interface TextModelConfig {
  numLayers: number;
  hiddenSize: number;
  intermediateSize: number;
  numHeads: number;
  numKVHeads: number;
  headDim: number;
  vocabSize: number;
  maxSeqLen: number;
  quantization: string;
}

/**
 * Inferred attention parameters from tensor shapes
 */
interface InferredAttentionParams {
  numHeads: number;
  numKVHeads: number;
  headDim: number;
}

/**
 * Model memory estimate
 */
export interface ModelEstimate {
  weightsBytes: number;
  kvCacheBytes: number;
  totalBytes: number;
  modelConfig: TextModelConfig;
}

/**
 * Progress callback event
 */
export interface LoadProgressEvent {
  stage: 'connecting' | 'manifest' | 'estimate' | 'warming' | 'downloading' | 'loading';
  message: string;
  estimate?: ModelEstimate;
}

/**
 * Generation options
 */
export interface GenerateOptions {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  stopTokens?: number[];
  stopSequences?: string[];
  onToken?: (token: string) => void;
}

/**
 * Chat message format
 */
export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

/**
 * Chat response format
 */
export interface ChatResponse {
  content: string;
  usage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
}

/**
 * DOPPLER capability flags type
 */
export interface DopplerCapabilitiesType {
  available: boolean;
  HAS_MEMORY64: boolean;
  HAS_SUBGROUPS: boolean;
  HAS_F16: boolean;
  IS_UNIFIED_MEMORY: boolean;
  TIER_LEVEL: number;
  TIER_NAME: string;
  MAX_MODEL_SIZE: number;
  initialized: boolean;
  currentModelId: string | null;
  kernelsWarmed: boolean;
  kernelsTuned: boolean;
  lastModelEstimate: ModelEstimate | null;
  bridgeClient?: ExtensionBridgeClient | null;
  localPath?: string | null;
}

/**
 * DOPPLER capability flags (populated at init)
 */
export const DopplerCapabilities: DopplerCapabilitiesType = {
  available: false,
  HAS_MEMORY64: false,
  HAS_SUBGROUPS: false,
  HAS_F16: false,
  IS_UNIFIED_MEMORY: false,
  TIER_LEVEL: 1,
  TIER_NAME: '',
  MAX_MODEL_SIZE: 0,
  initialized: false,
  currentModelId: null,
  kernelsWarmed: false,
  kernelsTuned: false,
  lastModelEstimate: null,
};

function extractTextModelConfig(manifest: RDRRManifest): TextModelConfig {
  const cfg = manifest?.config || (manifest as unknown as Record<string, unknown>)?.modelConfig || {};
  const textCfg = (cfg as Record<string, unknown>)?.text_config || cfg;
  const textConfig = textCfg as Record<string, unknown>;

  const hiddenSize = (textConfig.hidden_size || textConfig.n_embd || 4096) as number;

  // Try to get attention params from config, or infer from tensor shapes
  let numHeads = (textConfig.num_attention_heads || textConfig.n_head) as number | undefined;
  let numKVHeads = textConfig.num_key_value_heads as number | undefined;
  let headDim = textConfig.head_dim as number | undefined;

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
    numLayers: (textConfig.num_hidden_layers || textConfig.n_layer || 32) as number,
    hiddenSize,
    intermediateSize: (textConfig.intermediate_size || textConfig.n_inner || 14336) as number,
    numHeads,
    numKVHeads,
    headDim,
    vocabSize: (textConfig.vocab_size || 32000) as number,
    maxSeqLen: (textConfig.max_position_embeddings || textConfig.context_length || 4096) as number,
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

function estimateDequantizedWeightsBytes(manifest: RDRRManifest): number {
  const q = (manifest?.quantization || '').toUpperCase();
  const total = manifest?.totalSize || 0;
  if (q.startsWith('Q4')) {
    // Roughly 8x expansion when dequantized to f32.
    return total * 8;
  }
  return total;
}

// Current state
let pipeline: InferencePipeline | null = null;
let currentModelId: string | null = null;

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

const readOPFSFile = async (path: string): Promise<ArrayBuffer> => {
  const { dir, filename } = await resolveOPFSPath(path, false);
  const handle = await dir.getFileHandle(filename);
  const file = await handle.getFile();
  return file.arrayBuffer();
};

const writeOPFSFile = async (path: string, data: ArrayBuffer): Promise<void> => {
  const { dir, filename } = await resolveOPFSPath(path, true);
  const handle = await dir.getFileHandle(filename, { create: true });
  const writable = await handle.createWritable();
  await writable.write(data);
  await writable.close();
};

const fetchArrayBuffer = async (url: string): Promise<ArrayBuffer> => {
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
async function initDoppler(): Promise<boolean> {
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
async function loadModel(
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
    const { getDevice } = await import('../src/gpu/device.js');

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
        const m = await import('../src/storage/shard-manager.js');
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

    pipeline = await createPipeline(manifest as unknown as import('../src/inference/pipeline/config.js').Manifest, {
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
async function unloadModel(): Promise<void> {
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
async function loadLoRAAdapter(adapter: LoRAManifest | RDRRManifest | string): Promise<void> {
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
async function unloadLoRAAdapter(): Promise<void> {
  if (!pipeline) return;
  pipeline.setLoRAAdapter(null);
  log.info('DopplerProvider', 'LoRA adapter unloaded');
}

/**
 * Get active LoRA adapter name
 */
function getActiveLoRA(): string | null {
  const active = pipeline?.getActiveLoRA() || null;
  return active ? active.name : null;
}

/**
 * Generate text completion
 * @param prompt - Input prompt
 * @param options - Generation options
 * @returns Token stream
 */
async function* generate(prompt: string, options: GenerateOptions = {}): AsyncGenerator<string> {
  if (!pipeline) {
    throw new Error('No model loaded. Call loadModel() first.');
  }

  const {
    maxTokens = 256,
    temperature = 0.7,
    topP = 0.9,
    topK = 40,
    stopSequences = [],
    onToken = null,
  } = options;

  for await (const token of pipeline.generate(prompt, {
    maxTokens,
    temperature,
    topP,
    topK,
    stopSequences,
  })) {
    if (onToken) onToken(token);
    yield token;
  }
}

async function prefillKV(prompt: string, options: GenerateOptions = {}): Promise<KVCacheSnapshot> {
  if (!pipeline) {
    throw new Error('No model loaded. Call loadModel() first.');
  }
  // Extract pipeline-compatible options (exclude provider-specific onToken)
  const { onToken: _unused, ...pipelineOptions } = options;
  return pipeline.prefillKVOnly(prompt, pipelineOptions);
}

async function* generateWithPrefixKV(
  prefix: KVCacheSnapshot,
  prompt: string,
  options: GenerateOptions = {}
): AsyncGenerator<string> {
  if (!pipeline) {
    throw new Error('No model loaded. Call loadModel() first.');
  }
  // Extract pipeline-compatible options (exclude provider-specific onToken)
  const { onToken, ...pipelineOptions } = options;
  for await (const token of pipeline.generateWithPrefixKV(prefix, prompt, pipelineOptions)) {
    if (onToken) onToken(token);
    yield token;
  }
}

/**
 * Format chat messages for Gemma models.
 *
 * Gemma uses: <start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n{content}<end_of_turn>\n
 * Gemma does NOT support system role - system instructions are merged into first user message.
 *
 * @param messages - Chat messages
 * @returns Formatted prompt string
 */
function formatGemmaChat(messages: ChatMessage[]): string {
  const parts: string[] = [];
  let systemContent = '';

  // Extract system content (will be prepended to first user message)
  for (const m of messages) {
    if (m.role === 'system') {
      systemContent += (systemContent ? '\n\n' : '') + m.content;
    }
  }

  // Format user/assistant turns
  for (const m of messages) {
    if (m.role === 'system') continue; // Already extracted

    if (m.role === 'user') {
      // Prepend system content to first user message
      const content = systemContent
        ? `${systemContent}\n\n${m.content}`
        : m.content;
      systemContent = ''; // Only prepend once
      parts.push(`<start_of_turn>user\n${content}<end_of_turn>\n`);
    } else if (m.role === 'assistant') {
      parts.push(`<start_of_turn>model\n${m.content}<end_of_turn>\n`);
    }
  }

  // Add model turn prefix for generation
  parts.push('<start_of_turn>model\n');

  return parts.join('');
}

/**
 * Format chat messages for Llama 3 instruct models.
 *
 * @param messages - Chat messages
 * @returns Formatted prompt string
 */
function formatLlama3Chat(messages: ChatMessage[]): string {
  const parts: string[] = ['<|begin_of_text|>'];

  for (const m of messages) {
    if (m.role === 'system') {
      parts.push(`<|start_header_id|>system<|end_header_id|>\n\n${m.content}<|eot_id|>`);
    } else if (m.role === 'user') {
      parts.push(`<|start_header_id|>user<|end_header_id|>\n\n${m.content}<|eot_id|>`);
    } else if (m.role === 'assistant') {
      parts.push(`<|start_header_id|>assistant<|end_header_id|>\n\n${m.content}<|eot_id|>`);
    }
  }

  // Add assistant turn prefix for generation
  parts.push('<|start_header_id|>assistant<|end_header_id|>\n\n');

  return parts.join('');
}

/**
 * Format chat messages based on model type.
 *
 * @param messages - Chat messages
 * @returns Formatted prompt string
 */
function formatChatMessages(messages: ChatMessage[]): string {
  // Check model type from pipeline config
  const isGemma = pipeline?.modelConfig?.isGemma3;
  const isLlama3 = pipeline?.modelConfig?.isLlama3Instruct;

  if (isGemma) {
    return formatGemmaChat(messages);
  } else if (isLlama3) {
    return formatLlama3Chat(messages);
  }

  // Generic fallback format
  return messages
    .map((m) => {
      if (m.role === 'system') return `System: ${m.content}`;
      if (m.role === 'user') return `User: ${m.content}`;
      if (m.role === 'assistant') return `Assistant: ${m.content}`;
      return m.content;
    })
    .join('\n') + '\nAssistant:';
}

/**
 * Chat completion (matches LLM client interface)
 * @param messages - Chat messages
 * @param options - Generation options
 */
async function dopplerChat(messages: ChatMessage[], options: GenerateOptions = {}): Promise<ChatResponse> {
  // Format messages using model-aware template
  const prompt = formatChatMessages(messages);

  // Count prompt tokens using pipeline's tokenizer
  let promptTokens = 0;
  if (pipeline && pipeline.tokenizer) {
    try {
      const encoded = pipeline.tokenizer.encode(prompt);
      promptTokens = encoded.length;
    } catch (e) {
      log.warn('DopplerProvider', 'Failed to count prompt tokens', e);
    }
  }

  const tokens: string[] = [];
  for await (const token of generate(prompt, options)) {
    tokens.push(token);
  }

  return {
    content: tokens.join(''),
    usage: {
      promptTokens,
      completionTokens: tokens.length,
      totalTokens: promptTokens + tokens.length,
    },
  };
}

/**
 * Get list of available models
 */
async function getAvailableModels(): Promise<string[]> {
  return listModels();
}

/**
 * Get storage info
 */
async function getDopplerStorageInfo(): Promise<unknown> {
  // Provide quota + OPFS report
  return getStorageReport();
}

/**
 * Cleanup DOPPLER resources
 */
async function destroyDoppler(): Promise<void> {
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

/**
 * Provider interface for llm-client.js integration
 */
export interface DopplerProviderInterface {
  name: string;
  displayName: string;
  isLocal: boolean;
  init(): Promise<boolean>;
  loadModel(
    modelId: string,
    modelUrl?: string | null,
    onProgress?: ((event: LoadProgressEvent) => void) | null,
    localPath?: string | null
  ): Promise<boolean>;
  chat(messages: ChatMessage[], options?: GenerateOptions): Promise<ChatResponse>;
  stream(messages: ChatMessage[], options?: GenerateOptions): AsyncGenerator<string>;
  prefillKV(prompt: string, options?: GenerateOptions): Promise<KVCacheSnapshot>;
  generateWithPrefixKV(prefix: KVCacheSnapshot, prompt: string, options?: GenerateOptions): AsyncGenerator<string>;
  loadLoRAAdapter(adapter: LoRAManifest | RDRRManifest | string): Promise<void>;
  unloadLoRAAdapter(): Promise<void>;
  getActiveLoRA(): string | null;
  getCapabilities(): DopplerCapabilitiesType;
  getModels(): Promise<string[]>;
  destroy(): Promise<void>;
}

/**
 * Provider definition for llm-client.js
 */
export const DopplerProvider: DopplerProviderInterface = {
  name: 'doppler',
  displayName: 'DOPPLER',
  isLocal: true,

  async init(): Promise<boolean> {
    return initDoppler();
  },

  async loadModel(
    modelId: string,
    modelUrl?: string | null,
    onProgress?: ((event: LoadProgressEvent) => void) | null,
    localPath?: string | null
  ): Promise<boolean> {
    return loadModel(modelId, modelUrl ?? null, onProgress ?? null, localPath ?? null);
  },

  async chat(messages: ChatMessage[], options?: GenerateOptions): Promise<ChatResponse> {
    return dopplerChat(messages, options);
  },

  async *stream(messages: ChatMessage[], options?: GenerateOptions): AsyncGenerator<string> {
    const prompt = formatChatMessages(messages);
    for await (const token of generate(prompt, options)) {
      yield token;
    }
  },

  async prefillKV(prompt: string, options?: GenerateOptions): Promise<KVCacheSnapshot> {
    return prefillKV(prompt, options);
  },

  async *generateWithPrefixKV(
    prefix: KVCacheSnapshot,
    prompt: string,
    options?: GenerateOptions
  ): AsyncGenerator<string> {
    for await (const token of generateWithPrefixKV(prefix, prompt, options)) {
      yield token;
    }
  },

  async loadLoRAAdapter(adapter: LoRAManifest | string): Promise<void> {
    return loadLoRAAdapter(adapter);
  },

  async unloadLoRAAdapter(): Promise<void> {
    return unloadLoRAAdapter();
  },

  getActiveLoRA(): string | null {
    return getActiveLoRA();
  },

  getCapabilities(): DopplerCapabilitiesType {
    return DopplerCapabilities;
  },

  async getModels(): Promise<string[]> {
    return getAvailableModels();
  },

  async destroy(): Promise<void> {
    return destroyDoppler();
  },
};

export default DopplerProvider;
