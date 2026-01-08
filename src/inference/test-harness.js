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

import { initDevice, getDevice, getKernelCapabilities } from '../gpu/device.js';
import { parseManifest } from '../storage/rdrr-format.js';
import { createPipeline } from './pipeline.js';
import { log as debugLog } from '../debug/index.js';
import { DEFAULT_HOTSWAP_CONFIG } from '../config/schema/hotswap.schema.js';
import { getRuntimeConfig, setRuntimeConfig } from '../config/runtime.js';
import {
  fetchHotSwapManifest,
  verifyHotSwapManifest,
} from '../hotswap/manifest.js';
import { setHotSwapManifest } from '../hotswap/runtime.js';

/**
 * @typedef {import('../gpu/device.js').KernelCapabilities} KernelCapabilities
 * @typedef {import('../storage/rdrr-format.js').RDRRManifest} RDRRManifest
 * @typedef {import('./pipeline.js').Pipeline} Pipeline
 * @typedef {import('./pipeline/config.js').Manifest} Manifest
 * @typedef {import('../config/schema/index.js').RuntimeConfigSchema} RuntimeConfigSchema
 * @typedef {import('../config/schema/index.js').KernelPathSchema} KernelPathSchema
 * @typedef {import('../config/schema/hotswap.schema.js').HotSwapConfigSchema} HotSwapConfigSchema
 * @typedef {import('./test-harness.js').ModelInfo} ModelInfo
 * @typedef {import('./test-harness.js').RuntimeOverrides} RuntimeOverrides
 * @typedef {import('./test-harness.js').InferenceHarnessOptions} InferenceHarnessOptions
 * @typedef {import('./test-harness.js').InitializeResult} InitializeResult
 * @typedef {import('./test-harness.js').TestState} TestState
 */

// ============================================================================
// Model Discovery
// ============================================================================

/**
 * Discover available models from the /api/models endpoint.
 *
 * @param {string[]} [fallbackModels=['gemma3-1b-q4', 'mistral-7b-q4', 'llama3-8b-q4']] - Models to return if API fails
 * @returns {Promise<ModelInfo[]>} Array of model info objects
 */
export async function discoverModels(
  fallbackModels = ['gemma3-1b-q4', 'mistral-7b-q4', 'llama3-8b-q4']
) {
  try {
    const resp = await fetch('/api/models');
    if (resp.ok) {
      const models = await resp.json();
      return models.map((/** @type {ModelInfo | string} */ m) => {
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
 * - debug: Enable debug mode
 *
 * @param {URLSearchParams} [searchParams] - URLSearchParams to parse (default: window.location.search)
 * @returns {RuntimeOverrides} RuntimeOverrides object
 */
export function parseRuntimeOverridesFromURL(searchParams) {
  const params = searchParams || new URLSearchParams(window.location.search);

  /** @type {RuntimeOverrides} */
  const runtime = {};

  // Kernel path (new, preferred) - can be preset ID or inline JSON
  const kernelPathRaw = params.get('kernelPath');
  if (kernelPathRaw) {
    if (kernelPathRaw.startsWith('{')) {
      try {
        runtime.kernelPath = JSON.parse(kernelPathRaw);
      } catch (e) {
        debugLog.warn('TestHarness', `Failed to parse kernelPath JSON: ${/** @type {Error} */ (e).message}`);
      }
    } else {
      // Preset ID (e.g., 'gemma2-q4k-fused')
      runtime.kernelPath = kernelPathRaw;
    }
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
        runtime.runtimeConfig = /** @type {Partial<RuntimeConfigSchema>} */ (parsed);
      }
    } catch (e) {
      debugLog.warn('TestHarness', `Failed to parse runtimeConfig JSON: ${/** @type {Error} */ (e).message}`);
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
      debugLog.warn('TestHarness', `Failed to parse configChain JSON: ${/** @type {Error} */ (e).message}`);
    }
  }

  const hotSwapManifest = params.get('hotSwapManifest');
  const hotSwapLocalOnly = params.get('hotSwapLocalOnly');
  const hotSwapAllowUnsignedLocal = params.get('hotSwapAllowUnsignedLocal');
  if (hotSwapManifest || hotSwapLocalOnly || hotSwapAllowUnsignedLocal) {
    runtime.runtimeConfig = runtime.runtimeConfig ?? {};
    const baseHotSwap = getRuntimeConfig().hotSwap ?? DEFAULT_HOTSWAP_CONFIG;
    /** @type {Partial<HotSwapConfigSchema>} */
    const overrideHotSwap = runtime.runtimeConfig.hotSwap ?? {};
    /** @type {HotSwapConfigSchema} */
    const hotSwap = {
      ...baseHotSwap,
      ...overrideHotSwap,
      enabled: overrideHotSwap.enabled ?? baseHotSwap.enabled,
      trustedSigners: overrideHotSwap.trustedSigners ?? baseHotSwap.trustedSigners,
    };
    if (hotSwapManifest) {
      hotSwap.manifestUrl = hotSwapManifest;
      hotSwap.enabled = true;
    }
    if (hotSwapLocalOnly) {
      hotSwap.localOnly = hotSwapLocalOnly === '1' || hotSwapLocalOnly === 'true';
    }
    if (hotSwapAllowUnsignedLocal) {
      hotSwap.allowUnsignedLocal =
        hotSwapAllowUnsignedLocal === '1' || hotSwapAllowUnsignedLocal === 'true';
    }
    runtime.runtimeConfig.hotSwap = hotSwap;
  }

  return runtime;
}

// ============================================================================
// Shard Loading
// ============================================================================

/**
 * Create an HTTP-based shard loader for a model.
 *
 * @param {string} baseUrl - Base URL for the model (e.g., http://localhost:8080/doppler/models/gemma-1b-q4)
 * @param {RDRRManifest} manifest - Parsed model manifest
 * @param {(msg: string, level?: string) => void} [log] - Optional logging function
 * @returns {(idx: number) => Promise<Uint8Array>} Async function that loads a shard by index
 */
export function createHttpShardLoader(baseUrl, manifest, log) {
  const totalShards = manifest.shards?.length || 0;
  /** @type {Map<number, Uint8Array>} */
  const shardCache = new Map();
  /** @type {Map<number, Promise<Uint8Array>>} */
  const pendingLoads = new Map();
  let shardsLoaded = 0;
  let totalBytesLoaded = 0;
  const loadStartTime = Date.now();

  return async (/** @type {number} */ idx) => {
    const shard = manifest.shards[idx];
    if (!shard) {
      throw new Error(`No shard at index ${idx}`);
    }

    // Return cached shard if already loaded
    if (shardCache.has(idx)) {
      return /** @type {Uint8Array} */ (shardCache.get(idx));
    }

    // Wait for pending load if one is in progress (avoid duplicate fetches)
    if (pendingLoads.has(idx)) {
      return /** @type {Promise<Uint8Array>} */ (pendingLoads.get(idx));
    }

    // Start new load and track it as pending
    const shardStartTime = Date.now();
    /** @type {Promise<Uint8Array>} */
    const loadPromise = (async () => {
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
 * @param {string} manifestUrl - URL to manifest.json
 * @returns {Promise<RDRRManifest>} Parsed manifest
 */
export async function fetchManifest(manifestUrl) {
  const response = await fetch(manifestUrl);
  if (!response.ok) {
    throw new Error(`Failed to fetch manifest: ${response.status}`);
  }
  return parseManifest(await response.text());
}

/**
 * Initialize the WebGPU device and return capabilities.
 *
 * @returns {Promise<KernelCapabilities>} Kernel capabilities
 */
export async function initializeDevice() {
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
 * @param {string} modelUrl - Base URL for the model directory
 * @param {InferenceHarnessOptions} [options={}] - Initialization options
 * @returns {Promise<InitializeResult>} Pipeline and associated info
 */
export async function initializeInference(modelUrl, options = {}) {
  const log = options.log || ((/** @type {string} */ msg) => debugLog.info('TestHarness', msg));
  const onProgress = options.onProgress || (() => {});
  if (options.runtime?.runtimeConfig) {
    setRuntimeConfig(options.runtime.runtimeConfig);
  }

  const hotSwapConfig = getRuntimeConfig().hotSwap;
  if (hotSwapConfig.enabled && hotSwapConfig.manifestUrl) {
    onProgress('hotswap', 0.05, 'Loading hot-swap manifest...');
    log(`Hot-swap: loading manifest ${hotSwapConfig.manifestUrl}`);
    const hotSwapManifest = await fetchHotSwapManifest(hotSwapConfig.manifestUrl);
    const verification = await verifyHotSwapManifest(hotSwapManifest, hotSwapConfig);
    if (!verification.ok) {
      throw new Error(`Hot-swap manifest rejected: ${verification.reason}`);
    }
    setHotSwapManifest(hotSwapManifest);
    log(`Hot-swap manifest accepted: ${hotSwapManifest.bundleId} (${verification.reason})`);
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
  /** @type {RuntimeOverrides} */
  const runtime = {
    debug: true,
    ...options.runtime,
  };

  // 5. Create pipeline
  onProgress('pipeline', 0.2, 'Creating pipeline...');
  log('Creating pipeline...');

  const pipeline = await createPipeline(/** @type {Manifest} */ (/** @type {unknown} */ (manifest)), {
    storage: { loadShard },
    gpu: { device },
    baseUrl: modelUrl,
    runtime: {
      debug: runtime.debug,
      kernelPath: runtime.kernelPath,
    },
    onProgress: (/** @type {{ percent: number; stage?: string; message?: string }} */ progress) => {
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
 * Create initial test state object.
 * @returns {TestState}
 */
export function createTestState() {
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
