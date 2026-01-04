/**
 * test-harness.ts - Shared Inference Test Utilities
 *
 * Common utilities for inference testing and automation:
 * - Model discovery via /api/models
 * - URL parameter parsing for runtime overrides
 * - HTTP-based shard loading
 * - Pipeline initialization helpers
 *
 * Used by test-inference.html and potentially other test harnesses.
 *
 * @module inference/test-harness
 */

import { initDevice, getDevice, getKernelCapabilities, type KernelCapabilities } from '../gpu/device.js';
import { parseManifest, type RDRRManifest, type KernelHints } from '../storage/rdrr-format.js';
import { createPipeline, type Pipeline } from './pipeline.js';
import type { Manifest } from './pipeline/config.js';
import { log as debugLog } from '../debug/index.js';
import type { RuntimeConfigSchema } from '../config/schema/index.js';
import { setRuntimeConfig } from '../config/runtime.js';

// ============================================================================
// Types
// ============================================================================

/**
 * Model info returned from /api/models
 */
export interface ModelInfo {
  id: string;
  name: string;
  path?: string;
  numLayers?: number;
  vocabSize?: number;
  quantization?: string;
  downloadSize?: number;
  architecture?: string;
}

/**
 * Runtime overrides parsed from URL parameters
 */
export interface RuntimeOverrides {
  debug?: boolean;
  attentionKernel?: string;
  kernelHints?: KernelHints;
  runtimeConfig?: Partial<RuntimeConfigSchema>;
  /** Enable GPU timestamp profiling */
  profile?: boolean;
  /** Trace level: 'quick' | 'full' */
  trace?: string;
  /** Specific layers to debug checkpoint */
  debugLayers?: number[];
  /** Config inheritance chain for debugging (e.g., ['debug', 'default']) */
  configChain?: string[];
}

/**
 * Options for pipeline initialization
 */
export interface InferenceHarnessOptions {
  /** Base URL for model files (default: inferred from model URL) */
  baseUrl?: string;
  /** Runtime overrides for kernel selection */
  runtime?: RuntimeOverrides;
  /** Progress callback */
  onProgress?: (phase: string, progress: number, detail?: string) => void;
  /** Log function (default: debug log) */
  log?: (msg: string, level?: string) => void;
}

/**
 * Result of pipeline initialization
 */
export interface InitializeResult {
  pipeline: Pipeline;
  manifest: RDRRManifest;
  capabilities: KernelCapabilities;
}

// ============================================================================
// Model Discovery
// ============================================================================

/**
 * Discover available models from the /api/models endpoint.
 *
 * @param fallbackModels - Models to return if API fails
 * @returns Array of model info objects
 */
export async function discoverModels(
  fallbackModels: string[] = ['gemma3-1b-q4', 'mistral-7b-q4', 'llama3-8b-q4']
): Promise<ModelInfo[]> {
  try {
    const resp = await fetch('/api/models');
    if (resp.ok) {
      const models = await resp.json();
      return models.map((m: ModelInfo | string) => {
        if (typeof m === 'string') {
          return { id: m, name: m };
        }
        return {
          id: m.id || m.name || 'unknown',
          name: m.name || m.id || 'Unknown',
          ...m,
        };
      });
    }
  } catch (e) {
    // API not available, use fallback
  }
  return fallbackModels.map((id) => ({ id, name: id }));
}

// ============================================================================
// URL Parameter Parsing
// ============================================================================

/**
 * Parse runtime overrides from URL query parameters.
 *
 * Supported parameters:
 * - kernelHints: JSON object with kernel hints
 * - attentionKernel: Attention kernel override
 * - computePrecision: Compute precision (f16/f32/auto)
 * - q4kMatmul: Q4K matmul strategy
 * - f16Matmul: F16 matmul strategy
 * - attentionPrefill: Prefill attention kernel
 * - attentionDecode: Decode attention kernel
 * - debug: Enable debug mode
 *
 * @param searchParams - URLSearchParams to parse (default: window.location.search)
 * @returns RuntimeOverrides object
 */
export function parseRuntimeOverridesFromURL(
  searchParams?: URLSearchParams
): RuntimeOverrides {
  const params = searchParams || new URLSearchParams(window.location.search);
  const hints: KernelHints = {};

  // Parse JSON kernelHints if present
  const kernelHintsRaw = params.get('kernelHints');
  if (kernelHintsRaw) {
    try {
      const parsed = JSON.parse(kernelHintsRaw);
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        Object.assign(hints, parsed);
      }
    } catch (e) {
      debugLog.warn('TestHarness', `Failed to parse kernelHints JSON: ${(e as Error).message}`);
    }
  }

  // Parse individual hint parameters
  const computePrecision = params.get('computePrecision');
  const q4kMatmul = params.get('q4kMatmul');
  const f16Matmul = params.get('f16Matmul');
  const attentionPrefill = params.get('attentionPrefill');
  const attentionDecode = params.get('attentionDecode');

  if (computePrecision) hints.computePrecision = computePrecision as KernelHints['computePrecision'];
  if (q4kMatmul) hints.q4kMatmul = q4kMatmul as KernelHints['q4kMatmul'];
  if (f16Matmul) hints.f16Matmul = f16Matmul as KernelHints['f16Matmul'];
  if (attentionPrefill) hints.attentionPrefill = attentionPrefill as KernelHints['attentionPrefill'];
  if (attentionDecode) hints.attentionDecode = attentionDecode as KernelHints['attentionDecode'];

  const runtime: RuntimeOverrides = {};

  // Attention kernel override
  const attentionKernel = params.get('attentionKernel');
  if (attentionKernel) {
    runtime.attentionKernel = attentionKernel;
  }

  // Debug mode
  if (params.has('debug')) {
    runtime.debug = true;
  }

  // GPU profiling
  if (params.has('profile')) {
    runtime.profile = true;
  }

  // Trace level
  const trace = params.get('trace');
  if (trace) {
    runtime.trace = trace;
  }

  // Debug layers (comma-separated list of layer indices)
  const debugLayersStr = params.get('debugLayers');
  if (debugLayersStr) {
    runtime.debugLayers = debugLayersStr
      .split(',')
      .map(s => parseInt(s.trim(), 10))
      .filter(n => !isNaN(n));
  }

  // Runtime config (full or partial)
  const runtimeConfigRaw = params.get('runtimeConfig');
  if (runtimeConfigRaw) {
    try {
      const parsed = JSON.parse(runtimeConfigRaw);
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        runtime.runtimeConfig = parsed as Partial<RuntimeConfigSchema>;
      }
    } catch (e) {
      debugLog.warn('TestHarness', `Failed to parse runtimeConfig JSON: ${(e as Error).message}`);
    }
  }

  // Config chain (for debugging)
  const configChainRaw = params.get('configChain');
  if (configChainRaw) {
    try {
      const parsed = JSON.parse(configChainRaw);
      if (Array.isArray(parsed)) {
        runtime.configChain = parsed;
        debugLog.info('TestHarness', `Config chain: ${parsed.join(' → ')}`);
      }
    } catch (e) {
      debugLog.warn('TestHarness', `Failed to parse configChain JSON: ${(e as Error).message}`);
    }
  }

  // Attach hints if any were specified
  if (Object.keys(hints).length > 0) {
    runtime.kernelHints = hints;
  }

  return runtime;
}

// ============================================================================
// Shard Loading
// ============================================================================

/**
 * Create an HTTP-based shard loader for a model.
 *
 * @param baseUrl - Base URL for the model (e.g., http://localhost:8080/doppler/models/gemma-1b-q4)
 * @param manifest - Parsed model manifest
 * @param log - Optional logging function
 * @returns Async function that loads a shard by index
 */
export function createHttpShardLoader(
  baseUrl: string,
  manifest: RDRRManifest,
  log?: (msg: string, level?: string) => void
): (idx: number) => Promise<Uint8Array> {
  const totalShards = manifest.shards?.length || 0;
  const shardCache = new Map<number, Uint8Array>();
  const pendingLoads = new Map<number, Promise<Uint8Array>>();
  let shardsLoaded = 0;
  let totalBytesLoaded = 0;
  const loadStartTime = Date.now();

  return async (idx: number): Promise<Uint8Array> => {
    const shard = manifest.shards[idx];
    if (!shard) {
      throw new Error(`No shard at index ${idx}`);
    }

    // Return cached shard if already loaded
    if (shardCache.has(idx)) {
      return shardCache.get(idx)!;
    }

    // Wait for pending load if one is in progress (avoid duplicate fetches)
    if (pendingLoads.has(idx)) {
      return pendingLoads.get(idx)!;
    }

    // Start new load and track it as pending
    const shardStartTime = Date.now();
    const loadPromise = (async (): Promise<Uint8Array> => {
      const resp = await fetch(`${baseUrl}/${shard.filename}`);
      if (!resp.ok) {
        throw new Error(`Failed to load shard ${idx}: ${resp.status}`);
      }

      const data = new Uint8Array(await resp.arrayBuffer());
      shardCache.set(idx, data);
      pendingLoads.delete(idx);
      shardsLoaded++;
      totalBytesLoaded += data.byteLength;

      // Note: Individual shard progress is now reported through pipeline onProgress callback
      // to avoid noisy duplicate logging. Log summary only when all shards loaded.
      if (log && shardsLoaded === totalShards) {
        const totalElapsed = (Date.now() - loadStartTime) / 1000;
        const avgSpeed = totalElapsed > 0 ? totalBytesLoaded / totalElapsed : 0;
        log(`All ${totalShards} shards loaded: ${(totalBytesLoaded / 1024 / 1024).toFixed(1)}MB in ${totalElapsed.toFixed(1)}s (${(avgSpeed / 1024 / 1024).toFixed(0)} MB/s avg)`);
      }

      return data;
    })();

    pendingLoads.set(idx, loadPromise);
    return loadPromise;
  };
}

// ============================================================================
// Pipeline Initialization
// ============================================================================

/**
 * Fetch and parse a model manifest from a URL.
 *
 * @param manifestUrl - URL to manifest.json
 * @returns Parsed manifest
 */
export async function fetchManifest(manifestUrl: string): Promise<RDRRManifest> {
  const response = await fetch(manifestUrl);
  if (!response.ok) {
    throw new Error(`Failed to fetch manifest: ${response.status}`);
  }
  return parseManifest(await response.text());
}

/**
 * Initialize the WebGPU device and return capabilities.
 *
 * @returns Kernel capabilities
 */
export async function initializeDevice(): Promise<KernelCapabilities> {
  await initDevice();
  return getKernelCapabilities();
}

/**
 * Initialize a complete inference pipeline from a model URL.
 *
 * This is a convenience function that handles:
 * 1. WebGPU device initialization
 * 2. Manifest fetching and parsing
 * 3. Pipeline creation with shard loading
 *
 * @param modelUrl - Base URL for the model directory
 * @param options - Initialization options
 * @returns Pipeline and associated info
 */
export async function initializeInference(
  modelUrl: string,
  options: InferenceHarnessOptions = {}
): Promise<InitializeResult> {
  const log = options.log || ((msg: string) => debugLog.info('TestHarness', msg));
  const onProgress = options.onProgress || (() => {});
  if (options.runtime?.runtimeConfig) {
    setRuntimeConfig(options.runtime.runtimeConfig);
  }

  // 1. Initialize WebGPU
  onProgress('init', 0, 'Initializing WebGPU...');
  log('Initializing WebGPU...');

  await initDevice();
  const device = getDevice();
  const capabilities = getKernelCapabilities();

  log(`GPU: hasF16=${capabilities.hasF16}, hasSubgroups=${capabilities.hasSubgroups}`);

  // 2. Fetch manifest
  onProgress('manifest', 0.1, 'Fetching manifest...');
  log('Fetching manifest...');

  const manifestUrl = `${modelUrl}/manifest.json`;
  const manifest = await fetchManifest(manifestUrl);

  log(`Model: ${manifest.architecture || manifest.modelId || 'unknown'}`);

  // 3. Create shard loader
  const loadShard = createHttpShardLoader(modelUrl, manifest, log);

  // 4. Build runtime options
  const runtime: RuntimeOverrides = {
    debug: true,
    ...options.runtime,
  };

  // 5. Create pipeline
  onProgress('pipeline', 0.2, 'Creating pipeline...');
  log('Creating pipeline...');

  const pipeline = await createPipeline(manifest as unknown as Manifest, {
    storage: { loadShard },
    gpu: { device },
    baseUrl: modelUrl,
    runtime: {
      attentionKernel: runtime.attentionKernel || 'auto',
      debug: runtime.debug,
      kernelHints: runtime.kernelHints,
    },
    onProgress: (progress: { percent: number; stage?: string; message?: string }) => {
      const pct = 0.2 + progress.percent * 0.8;
      onProgress(progress.stage || 'loading', pct, progress.message);
    },
  });

  onProgress('complete', 1, 'Ready');
  log('Pipeline ready');

  return { pipeline, manifest, capabilities };
}

// ============================================================================
// Test State (for Playwright automation)
// ============================================================================

/**
 * Standard test state interface for Playwright automation.
 */
export interface TestState {
  ready: boolean;
  loading: boolean;
  loaded: boolean;
  generating: boolean;
  done: boolean;
  output: string;
  tokens: string[];
  errors: string[];
  model: string | null;
}

/**
 * Create initial test state object.
 */
export function createTestState(): TestState {
  return {
    ready: false,
    loading: false,
    loaded: false,
    generating: false,
    done: false,
    output: '',
    tokens: [],
    errors: [],
    model: null,
  };
}
