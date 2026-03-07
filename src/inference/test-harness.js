

import { initDevice, getDevice, getKernelCapabilities } from '../gpu/device.js';
import { parseManifest } from '../formats/rdrr/index.js';
import { createPipeline } from './pipelines/text.js';
import { log as debugLog } from '../debug/index.js';
import { getRuntimeConfig, setRuntimeConfig } from '../config/runtime.js';
import { downloadShard as downloadShardFromDistribution } from '../distribution/shard-delivery.js';
import {
  fetchHotSwapManifest,
  verifyHotSwapManifest,
} from '../hotswap/manifest.js';
import { evaluateHotSwapRollout, setHotSwapManifest } from '../hotswap/runtime.js';
import {
  fetchIntentBundle,
  getKernelRegistryVersion,
  verifyIntentBundle,
} from '../hotswap/intent-bundle.js';



// ============================================================================
// Model Discovery
// ============================================================================


export async function discoverModels(
  fallbackModels
) {
  try {
    const resp = await fetch('/models/catalog.json');
    if (resp.ok) {
      const payload = await resp.json();
      const catalogModels = Array.isArray(payload.models) ? payload.models : [];
      if (catalogModels.length > 0) {
        return catalogModels.map((m) => ({
          id: m.modelId || m.id || 'unknown',
          name: m.label || m.name || m.modelId || 'Unknown',
          ...m,
        }));
      }
    }
  } catch (e) {}

  if (Array.isArray(fallbackModels) && fallbackModels.length > 0) {
    return fallbackModels.map((id) => ({ id, name: id }));
  }

  throw new Error('discoverModels: failed to fetch /models/catalog.json and no explicit fallback model list was provided.');
}

// ============================================================================
// URL Parameter Parsing
// ============================================================================


export function parseRuntimeOverridesFromURL(searchParams) {
  const query = typeof globalThis.location !== 'undefined' ? globalThis.location.search : '';
  const params = searchParams || new URLSearchParams(query);

  
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

function buildManifestVersionSet(manifest) {
  if (!manifest || typeof manifest !== 'object') return 'manifest:invalid';
  const shards = Array.isArray(manifest.shards)
    ? manifest.shards.map((shard, index) => ({
      index,
      filename: shard?.filename ?? null,
      size: shard?.size ?? null,
      hash: shard?.hash ?? null,
    }))
    : [];
  const payload = {
    modelId: manifest.modelId ?? null,
    version: manifest.version ?? null,
    hashAlgorithm: manifest.hashAlgorithm ?? null,
    tensorCount: manifest.tensorCount ?? null,
    totalSize: manifest.totalSize ?? null,
    shards,
  };
  return JSON.stringify(payload);
}

function toShardBytes(buffer, shardIndex) {
  if (buffer instanceof ArrayBuffer) {
    return new Uint8Array(buffer);
  }
  if (ArrayBuffer.isView(buffer)) {
    const view = buffer;
    return new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
  }
  throw new Error(`Shard ${shardIndex} did not return an in-memory buffer`);
}


export function createHttpShardLoader(baseUrl, manifest, log) {
  const algorithm = manifest.hashAlgorithm;
  if (!algorithm) {
    throw new Error('Manifest missing hashAlgorithm for shard delivery.');
  }

  const runtimeConfig = getRuntimeConfig();
  const distributionConfig = runtimeConfig.loading?.distribution || {};
  const totalShards = manifest.shards?.length || 0;
  const requiredEncoding = distributionConfig.requiredContentEncoding ?? null;
  const manifestVersionSet = buildManifestVersionSet(manifest);
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
    const loadPromise = (async () => {
      try {
        const result = await downloadShardFromDistribution(baseUrl, idx, shard, {
          distributionConfig,
          algorithm,
          requiredEncoding,
          expectedHash: shard.hash ?? null,
          expectedSize: Number.isFinite(shard.size) ? Math.floor(shard.size) : null,
          expectedManifestVersionSet: manifestVersionSet,
          writeToStore: false,
          enableSourceCache: true,
        });

        const data = toShardBytes(result.buffer, idx);
        shardCache.set(idx, data);
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
      } finally {
        pendingLoads.delete(idx);
      }
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
    const rolloutDecision = evaluateHotSwapRollout(hotSwapConfig, {
      modelUrl,
      subjectId: options.modelId || null,
      sessionId: options.sessionId || null,
      optInTag: options.hotSwapOptInTag || null,
    });
    if (!rolloutDecision.allowed) {
      log(`Hot-swap: rollout skipped (${rolloutDecision.reason})`);
    } else {
      onProgress('hotswap', 0.05, 'Loading hot-swap manifest...');
      log(`Hot-swap: loading manifest ${hotSwapConfig.manifestUrl}`);
      const hotSwapManifest = await fetchHotSwapManifest(hotSwapConfig.manifestUrl);
      const verification = await verifyHotSwapManifest(hotSwapManifest, hotSwapConfig, {
        source: {
          kind: 'remote',
          isLocal: false,
          url: hotSwapConfig.manifestUrl,
        },
      });
      if (!verification.ok) {
        throw new Error(`Hot-swap manifest rejected: ${verification.reason}`);
      }
      setHotSwapManifest(hotSwapManifest);
      log(
        `Hot-swap manifest accepted: ${hotSwapManifest.bundleId} (${verification.reason}, rollout=${rolloutDecision.reason})`
      );
    }
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
