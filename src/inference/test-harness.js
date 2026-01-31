

import { initDevice, getDevice, getKernelCapabilities } from '../gpu/device.js';
import { parseManifest } from '../storage/rdrr-format.js';
import { createPipeline } from './pipeline.js';
import { log as debugLog } from '../debug/index.js';
import { getRuntimeConfig, setRuntimeConfig } from '../config/runtime.js';
import {
  fetchHotSwapManifest,
  verifyHotSwapManifest,
} from '../hotswap/manifest.js';
import { setHotSwapManifest } from '../hotswap/runtime.js';
import {
  fetchIntentBundle,
  getKernelRegistryVersion,
  verifyIntentBundle,
} from '../hotswap/intent-bundle.js';



// ============================================================================
// Model Discovery
// ============================================================================


export async function discoverModels(
  fallbackModels = ['gemma3-1b-q4', 'mistral-7b-q4', 'llama3-8b-q4']
) {
  try {
    const resp = await fetch('/api/models');
    if (resp.ok) {
      const models = await resp.json();
      return models.map(( m) => {
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


export function parseRuntimeOverridesFromURL(searchParams) {
  const params = searchParams || new URLSearchParams(window.location.search);

  
  const runtime = {};

  // Runtime config (full or partial)
  const runtimeConfigRaw = params.get('runtimeConfig');
  if (runtimeConfigRaw) {
    try {
      const parsed = JSON.parse(runtimeConfigRaw);
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        runtime.runtimeConfig =  (parsed);
      }
    } catch (e) {
      debugLog.warn('TestHarness', `Failed to parse runtimeConfig JSON: ${ (e).message}`);
    }
  }

  // Config chain (for debugging)
  const configChainRaw = params.get('configChain');
  if (configChainRaw) {
    try {
      const parsed = JSON.parse(configChainRaw);
      if (Array.isArray(parsed)) {
        runtime.configChain = parsed;
        debugLog.info('TestHarness', `Config chain: ${parsed.join(' -> ')}`);
      }
    } catch (e) {
      debugLog.warn('TestHarness', `Failed to parse configChain JSON: ${ (e).message}`);
    }
  }

  return runtime;
}

// ============================================================================
// Shard Loading
// ============================================================================

function normalizeContentEncodings(value) {
  if (!value) return [];
  return value
    .split(',')
    .map((entry) => entry.trim().toLowerCase())
    .filter(Boolean);
}

function assertRequiredContentEncoding(response, requiredEncoding, context) {
  if (!requiredEncoding) return;
  const required = requiredEncoding.trim().toLowerCase();
  if (!required) return;
  const encodings = normalizeContentEncodings(response.headers.get('content-encoding'));
  if (!encodings.includes(required)) {
    const found = encodings.length > 0 ? encodings.join(', ') : 'none';
    throw new Error(`Missing required content-encoding "${required}" for ${context} (found: ${found})`);
  }
}


export function createHttpShardLoader(baseUrl, manifest, log) {
  const totalShards = manifest.shards?.length || 0;
  const requiredEncoding = getRuntimeConfig().loading.distribution.requiredContentEncoding;
  
  const shardCache = new Map();
  
  const pendingLoads = new Map();
  let shardsLoaded = 0;
  let totalBytesLoaded = 0;
  const loadStartTime = Date.now();

  return async ( idx) => {
    const shard = manifest.shards[idx];
    if (!shard) {
      throw new Error(`No shard at index ${idx}`);
    }

    // Return cached shard if already loaded
    if (shardCache.has(idx)) {
      return  (shardCache.get(idx));
    }

    // Wait for pending load if one is in progress (avoid duplicate fetches)
    if (pendingLoads.has(idx)) {
      return  (pendingLoads.get(idx));
    }

    // Start new load and track it as pending
    const shardStartTime = Date.now();
    
    const loadPromise = (async () => {
      const resp = await fetch(`${baseUrl}/${shard.filename}`);
      if (!resp.ok) {
        throw new Error(`Failed to load shard ${idx}: ${resp.status}`);
      }
      assertRequiredContentEncoding(resp, requiredEncoding, `shard ${idx}`);

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


export async function fetchManifest(manifestUrl) {
  const response = await fetch(manifestUrl);
  if (!response.ok) {
    throw new Error(`Failed to fetch manifest: ${response.status}`);
  }
  return parseManifest(await response.text());
}


export async function initializeDevice() {
  await initDevice();
  return getKernelCapabilities();
}


export async function initializeInference(modelUrl, options = {}) {
  const log = options.log || (( msg) => debugLog.info('TestHarness', msg));
  const onProgress = options.onProgress || (() => {});
  if (options.runtime?.runtimeConfig) {
    setRuntimeConfig(options.runtime.runtimeConfig);
  }

  const hotSwapConfig = getRuntimeConfig().shared.hotSwap;
  const intentBundleConfig = getRuntimeConfig().shared.intentBundle;
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

  if (intentBundleConfig.enabled && intentBundleConfig.bundleUrl) {
    onProgress('intent', 0.12, 'Loading intent bundle...');
    log(`Intent bundle: loading ${intentBundleConfig.bundleUrl}`);
    const bundle = await fetchIntentBundle(intentBundleConfig.bundleUrl);
    const kernelRegistryVersion = intentBundleConfig.requireKernelRegistryVersion
      ? await getKernelRegistryVersion()
      : null;
    const verification = await verifyIntentBundle(bundle, {
      manifest: intentBundleConfig.requireBaseModelHash ? manifest : null,
      kernelRegistryVersion,
      enforceDeterministicOutput: intentBundleConfig.enforceDeterministicOutput,
    });
    if (!verification.ok) {
      const reason = verification.reasons?.length
        ? `${verification.reason}: ${verification.reasons.join('; ')}`
        : verification.reason;
      throw new Error(`Intent bundle rejected: ${reason}`);
    }
    log(`Intent bundle accepted (${verification.reason})`);
    intentBundleConfig.bundle = bundle;
  }

  const modelLabel = typeof manifest.architecture === 'string'
    ? manifest.architecture
    : (manifest.modelType || manifest.modelId || 'unknown');
  log(`Model: ${modelLabel}`);

  // 3. Create shard loader
  const loadShard = createHttpShardLoader(modelUrl, manifest, log);

  // 4. Build runtime options
  
  const runtime = {
    ...options.runtime,
  };

  // 5. Create pipeline
  onProgress('pipeline', 0.2, 'Creating pipeline...');
  log('Creating pipeline...');

  const pipeline = await createPipeline( ( (manifest)), {
    storage: { loadShard },
    gpu: { device },
    baseUrl: modelUrl,
    onProgress: ( progress) => {
      const pct = 0.2 + progress.percent * 0.8;
      onProgress(progress.stage || 'loading', pct, progress.message);
    },
  });

  onProgress('complete', 1, 'Ready');
  log('Pipeline ready');

  // Snapshot active configuration for diffing
  const configSnapshot = {
     kernelPathId: pipeline.resolvedKernelPath?.id || null,
     kernelPathName: pipeline.resolvedKernelPath?.name || null,
     activeOverrides: options.runtime?.runtimeConfig?.inference?.kernelOverrides || null,
     // Detailed per-op view could be expanded here if needed
  };

  return { pipeline, manifest, capabilities, configSnapshot };
}

// ============================================================================
// Test State (for browser automation)
// ============================================================================


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
