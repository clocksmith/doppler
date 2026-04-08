import { initializeInference } from './test-harness.js';
import { setRuntimeConfig } from '../config/runtime.js';
import { initDevice, getKernelCapabilities, getDevice } from '../gpu/device.js';
import { createPipeline } from './pipelines/text.js';
import { parseModelConfigFromManifest } from './pipelines/text/config.js';
import { resolveKernelPathState, activateKernelPathState } from './pipelines/text/model-load.js';
import { openModelStore, loadManifestFromStore } from '../storage/shard-manager.js';
import { parseManifest } from '../formats/rdrr/index.js';
import { cloneRuntimeConfig, resolveRuntime } from './browser-harness-runtime-helpers.js';
import { normalizeLoadMode } from './browser-harness-suite-helpers.js';
import { buildSourceArtifactFingerprint, createStoredSourceArtifactContext } from '../storage/source-artifact-store.js';

const NODE_SOURCE_RUNTIME_MODULE_PATH = '../tooling/node-source-runtime.js';

function isNodeRuntime() {
  return typeof process !== 'undefined' && !!process.versions?.node;
}

function resolveSourceVerifyHashes(options = {}) {
  const explicit = options?.runtime?.runtimeConfig?.loading?.shardCache?.verifyHashes;
  if (explicit == null) {
    return true;
  }
  return explicit === true;
}

function applyResolvedResidentBudget(runtime, maxResidentBytes) {
  if (!runtime || !runtime.runtimeConfig || !Number.isFinite(maxResidentBytes) || maxResidentBytes <= 0) {
    return runtime;
  }
  const runtimeConfig = cloneRuntimeConfig(runtime.runtimeConfig);
  if (!runtimeConfig?.loading?.memoryManagement?.budget) {
    throw new Error(
      'loadMode=memory requires runtime.loading.memoryManagement.budget to resolve the resident memory budget.'
    );
  }
  runtimeConfig.loading.memoryManagement.budget.maxResidentBytes = Math.floor(maxResidentBytes);
  return {
    ...runtime,
    runtimeConfig,
  };
}

export function resolveDeviceInfo() {
  try {
    return getKernelCapabilities();
  } catch {
    return null;
  }
}

export async function resolveKernelPathForModel(options = {}) {
  const runtimeConfig = options.runtime?.runtimeConfig ?? null;
  let manifest = null;
  let manifestModelId = options.modelId || null;

  if (options.modelId) {
    await openModelStore(options.modelId);
    const manifestText = await loadManifestFromStore();
    if (manifestText) {
      manifest = parseManifest(manifestText);
      manifestModelId = manifest.modelId ?? options.modelId;
    }
  }

  if (!manifest) return null;

  const modelConfig = parseModelConfigFromManifest(manifest, runtimeConfig);
  const kernelPathState = resolveKernelPathState({
    manifest,
    runtimeConfig,
    modelConfig,
  });
  activateKernelPathState(kernelPathState);
  return {
    modelId: manifestModelId,
    kernelPath: kernelPathState.resolvedKernelPath,
    source: kernelPathState.kernelPathSource,
  };
}

export async function initializeInferenceFromStorage(modelId, options = {}) {
  const { onProgress } = options;
  if (!modelId) {
    throw new Error('modelId is required');
  }

  if (options.runtime?.runtimeConfig) {
    setRuntimeConfig(options.runtime.runtimeConfig);
  }

  onProgress?.('storage', 0.05, 'Opening model store...');
  await openModelStore(modelId);

  onProgress?.('manifest', 0.1, 'Loading manifest...');
  const manifestText = await loadManifestFromStore();
  if (!manifestText) {
    throw new Error('Manifest not found in storage');
  }
  const manifest = parseManifest(manifestText);

  onProgress?.('gpu', 0.2, 'Initializing WebGPU...');
  await initDevice();
  const device = getDevice();
  const capabilities = getKernelCapabilities();

  onProgress?.('pipeline', 0.3, 'Creating pipeline...');
  const storage = buildSourceArtifactFingerprint(manifest)
    ? createStoredSourceArtifactContext(manifest, { verifyHashes: true })
    : null;
  const pipeline = await createPipeline(manifest, {
    gpu: { device },
    runtime: options.runtime,
    ...(storage ? { storage } : {}),
    onProgress,
  });

  return { pipeline, manifest, capabilities };
}

export async function initializeInferenceFromSourcePath(sourcePath, options = {}) {
  const { onProgress } = options;
  if (!sourcePath || typeof sourcePath !== 'string') {
    throw new Error('modelUrl is required for loadMode=memory.');
  }
  if (!isNodeRuntime()) {
    throw new Error('loadMode=memory source runtime is currently supported on Node only.');
  }
  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(sourcePath)) {
    throw new Error(
      'loadMode=memory expects a local filesystem path (Safetensors directory or .gguf file), not an URL.'
    );
  }

  onProgress?.('source', 0.05, 'Preparing source runtime bundle...');
  const { resolveNodeSourceRuntimeBundle } = await import(NODE_SOURCE_RUNTIME_MODULE_PATH);
  const sourceBundle = await resolveNodeSourceRuntimeBundle({
    inputPath: sourcePath,
    modelId: options.modelId || null,
    verifyHashes: resolveSourceVerifyHashes(options),
    runtimeConfig: options.runtime?.runtimeConfig ?? null,
  });
  if (!sourceBundle) {
    throw new Error(
      `No source-runtime model detected at "${sourcePath}". ` +
      'Expected a Safetensors directory or a .gguf file path.'
    );
  }
  const effectiveRuntime = applyResolvedResidentBudget(
    options.runtime,
    sourceBundle.resolvedMemoryBudgetBytes
  );
  if (effectiveRuntime?.runtimeConfig) {
    setRuntimeConfig(effectiveRuntime.runtimeConfig);
  }

  onProgress?.('gpu', 0.2, 'Initializing WebGPU...');
  await initDevice();
  const device = getDevice();
  const capabilities = getKernelCapabilities();

  onProgress?.('pipeline', 0.3, 'Creating pipeline...');
  const pipeline = await createPipeline(sourceBundle.manifest, {
    gpu: { device },
    runtime: effectiveRuntime,
    storage: sourceBundle.storageContext,
    onProgress,
  });

  return {
    pipeline,
    manifest: sourceBundle.manifest,
    capabilities,
  };
}

export async function resolveHarnessOverride(options = {}) {
  const input = typeof options.harnessOverride === 'function'
    ? await options.harnessOverride(options)
    : options.harnessOverride;

  if (!input || typeof input !== 'object') {
    throw new Error('harnessOverride must resolve to an object.');
  }

  if (!input.pipeline || typeof input.pipeline.generate !== 'function') {
    throw new Error('harnessOverride.pipeline.generate(request) is required.');
  }

  const manifest = input.manifest && typeof input.manifest === 'object'
    ? input.manifest
    : {
      modelId: options.modelId || 'diffusion-harness-override',
      modelType: 'diffusion',
    };

  const modelLoadMs = Number.isFinite(input.modelLoadMs)
    ? Math.max(0, input.modelLoadMs)
    : 0;

  return {
    ...input,
    manifest,
    modelLoadMs,
  };
}

export async function initializeSuiteModel(options = {}) {
  if (options.harnessOverride) {
    if (options.runtime?.runtimeConfig) {
      setRuntimeConfig(options.runtime.runtimeConfig);
    }
    return resolveHarnessOverride(options);
  }
  const loadStart = performance.now();
  const runtime = resolveRuntime(options);
  const loadMode = normalizeLoadMode(options.loadMode, !!options.modelUrl, options.modelUrl);
  let harness;
  if (loadMode === 'memory') {
    if (!options.modelUrl) {
      throw new Error('loadMode=memory requires modelUrl to be a local model path.');
    }
    harness = await initializeInferenceFromSourcePath(options.modelUrl, { ...options, runtime });
  } else if (options.modelId && !options.modelUrl) {
    harness = await initializeInferenceFromStorage(options.modelId, { ...options, runtime });
  } else {
    if (!options.modelUrl) {
      throw new Error('modelUrl is required for this suite');
    }
    harness = await initializeInference(options.modelUrl, {
      runtime,
      onProgress: options.onProgress,
      log: options.log,
    });
  }
  const modelLoadMs = Math.max(0, performance.now() - loadStart);
  return { ...harness, modelLoadMs };
}
