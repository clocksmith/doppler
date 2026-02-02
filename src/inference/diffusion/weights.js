import { getRuntimeConfig } from '../../config/runtime.js';
import { log, trace } from '../../debug/index.js';
import { getKernelCapabilities } from '../../gpu/device.js';
import { releaseBuffer } from '../../memory/buffer-pool.js';
import { createShardCache } from '../../loader/shard-cache.js';
import { buildTensorLocations } from '../../loader/shard-resolver.js';
import { assembleShardData } from '../../loader/tensor-reader.js';
import { loadTensorToGPU, loadTensorToCPU } from '../../loader/tensor-loader.js';
import { initStorage, openModelStore } from '../../storage/shard-manager.js';
import { setManifest } from '../../formats/rdrr/parsing.js';

function normalizeBaseUrl(baseUrl) {
  if (!baseUrl) return null;
  return baseUrl.replace(/\/$/, '');
}

function createRemoteShardLoader(baseUrl, manifest) {
  const root = normalizeBaseUrl(baseUrl);
  return async (shardIndex) => {
    const shard = manifest?.shards?.[shardIndex];
    const filename = shard?.filename;
    if (!filename) {
      throw new Error(`Shard ${shardIndex} missing filename in manifest.`);
    }
    const url = `${root}/${filename}`;
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch shard ${shardIndex}: ${response.status}`);
    }
    return response.arrayBuffer();
  };
}

function createLoaderConfig(runtime, manifest) {
  const caps = getKernelCapabilities();
  const keepF32Weights = runtime?.inference?.compute?.keepF32Weights === true;
  const allowF32UpcastNonMatmul = runtime?.loading?.allowF32UpcastNonMatmul !== false;
  return {
    gpuCapabilities: caps,
    useFusedQ4K: false,
    q4kLayout: manifest?.quantizationInfo?.layout ?? null,
    keepF32Weights,
    allowF32UpcastNonMatmul,
  };
}

export async function createDiffusionWeightLoader(manifest, options = {}) {
  if (!manifest) {
    throw new Error('Diffusion weight loader requires a manifest.');
  }

  const runtime = options.runtimeConfig ?? getRuntimeConfig();
  const loadingConfig = runtime.loading ?? getRuntimeConfig().loading;

  const shardCache = createShardCache(
    loadingConfig.shardCache.opfsEntries,
    loadingConfig.shardCache
  );

  if (options.baseUrl) {
    shardCache.setCustomLoader(createRemoteShardLoader(options.baseUrl, manifest), options.verifyHashes !== false);
  } else {
    await initStorage();
    await openModelStore(manifest.modelId);
  }

  shardCache.configureForModel(manifest, shardCache.hasCustomLoader);
  shardCache.setManifest(manifest);
  setManifest(manifest);

  const tensorsJsonUrl = manifest.tensorsFile && options.baseUrl
    ? `${normalizeBaseUrl(options.baseUrl)}/${manifest.tensorsFile}`
    : null;

  const tensorLocations = await buildTensorLocations(manifest, {
    hasCustomLoader: shardCache.hasCustomLoader,
    tensorsJsonUrl,
  });

  const loaderConfig = createLoaderConfig(runtime, manifest);

  const loadTensor = async (name, toGPU = true) => {
    const location = tensorLocations.get(name);
    if (!location) return null;

    const shardData = await assembleShardData(location, name, (idx) => shardCache.load(idx));

    if (toGPU) {
      const result = await loadTensorToGPU(shardData, location, name, loaderConfig);
      return {
        value: result.data,
        location,
        buffers: result.allocatedBuffers ?? [],
      };
    }

    const result = await loadTensorToCPU(shardData, location, name);
    return {
      value: result.data,
      location,
      buffers: result.allocatedBuffers ?? [],
    };
  };

  const loadComponentWeights = async (componentId, options = {}) => {
    const prefix = `${componentId}.`;
    const weights = new Map();
    const shapes = new Map();
    const dtypes = new Map();
    const buffers = new Set();
    const filter = options.filter ?? null;

    for (const [name, location] of tensorLocations.entries()) {
      if (!name.startsWith(prefix)) continue;
      if (filter && !filter(name, location)) continue;
      const result = await loadTensor(name, options.toGPU !== false);
      if (!result || !result.value) continue;
      weights.set(name, result.value);
      shapes.set(name, location.shape);
      dtypes.set(name, location.dtype);
      for (const buffer of result.buffers) {
        buffers.add(buffer);
      }
    }

    trace.loader(`Diffusion weights loaded: ${componentId} (${weights.size} tensors)`);

    const release = () => {
      for (const buffer of buffers) {
        releaseBuffer(buffer);
      }
    };

    return { weights, shapes, dtypes, release };
  };

  log.info('Diffusion', `Weight loader ready (${tensorLocations.size} tensors)`);

  return {
    tensorLocations,
    shardCache,
    loadTensor,
    loadComponentWeights,
  };
}
