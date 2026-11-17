import { createDopplerLoader } from "../../../../loader/doppler-loader.js";
import { log } from "../../../../debug/index.js";
import { resolvePerLayerInputsSession } from "../generator/session/session-context.js";
import { createHttpArtifactStorageContext, createNodeFileArtifactStorageContext } from "../../../../storage/artifact-storage-context.js";
import { resolveShardHashVerification } from "./load-policy.js";
import { rewriteWeightLoadError } from "./load-errors.js";
import { resolveWeightLoadingConfig } from "./weight-loading-config.js";
import { toArrayBuffer } from "../../../../formats/array-buffer.js";

function toUint8Array(value, label) {
  if (value instanceof Uint8Array) {
    return value;
  }
  if (value instanceof ArrayBuffer) {
    return new Uint8Array(value);
  }
  throw new Error(`${label} must return ArrayBuffer or Uint8Array.`);
}

function createStorageContextLoadError(error, operation, index, context = {}) {
  const message = error?.message || String(error);
  const wrapped = new Error(
    `storageContext.${operation} failed for shard ${index}: ${message}`,
    error instanceof Error ? { cause: error } : undefined
  );
  wrapped.name = error?.name || 'Error';
  if (error?.code !== undefined) {
    wrapped.code = error.code;
  }
  wrapped.details = {
    ...(error?.details && typeof error.details === 'object' ? error.details : {}),
    storageOperation: operation,
    shardIndex: index,
    ...context,
  };
  return wrapped;
}

function isRDRRManifest(manifest) {
  return manifest !== null && typeof manifest === 'object' && Array.isArray( (manifest).shards);
}

export function createNodeFileShardStorageContext(baseUrl, manifest, options = {}) {
  return createNodeFileArtifactStorageContext(baseUrl, manifest, options);
}

function createRemoteStorageContext(baseUrl, manifest, options = {}) {
  const nodeFileStorageContext = createNodeFileArtifactStorageContext(baseUrl, manifest, options);
  if (nodeFileStorageContext) {
    return nodeFileStorageContext;
  }
  return createHttpArtifactStorageContext(baseUrl, manifest, options);
}

// ============================================================================
// Weight Loading
// ============================================================================

export async function loadWeights(manifest, modelConfig, options = {}) {
  const {
    onProgress,
    loadingConfig,
    baseUrl,
    loaderDebug,
    perLayerInputSession,
  } = options;
  // Resolve the policy before creating a context so the context it creates
  // carries the same decision; a context must never shadow the configured policy.
  const configuredVerifyHashes = resolveShardHashVerification({ options, loadingConfig }).verifyHashes;
  const runtimeStorageContext = options.storageContext
    ?? createRemoteStorageContext(baseUrl, manifest, { verifyHashes: configuredVerifyHashes });
  if (typeof runtimeStorageContext?.preflight === 'function') {
    await runtimeStorageContext.preflight();
  }
  const { verifyHashes } = resolveShardHashVerification({
    options,
    loadingConfig,
    storageContext: runtimeStorageContext,
  });

  const dopplerLoader = options.loader ?? createDopplerLoader(loadingConfig);
  const keepF32Weights = options.keepF32Weights === true;
  dopplerLoader.setQ4KConfig({
    representationPaths: options.reachableKernelPaths ?? [options.resolvedKernelPath].filter(Boolean),
    ...resolveWeightLoadingConfig(
      manifest,
      options.resolvedKernelPath ?? null,
      options.kernelPathSource ?? 'none',
      keepF32Weights
    ),
  });
  dopplerLoader.setLoaderDebugConfig(loaderDebug ?? null);
  dopplerLoader.setPerLayerInputSession(
    resolvePerLayerInputsSession(
      modelConfig.perLayerInputsSession ?? null,
      perLayerInputSession ?? null
    )
  );

  const tensorsFile = isRDRRManifest(manifest) ? manifest.tensorsFile : null;
  if (baseUrl && tensorsFile) {
    const base = baseUrl.replace(/\/$/, '');
    const filename = tensorsFile.replace(/^\/+/, '');
    dopplerLoader.setTensorsJsonUrl(`${base}/${filename}`);
  } else {
    dopplerLoader.setTensorsJsonUrl(null);
  }
  dopplerLoader.setTensorsJsonLoader(
    typeof runtimeStorageContext?.loadTensorsJson === 'function'
      ? () => runtimeStorageContext.loadTensorsJson()
      : null
  );

  // Configure custom shard loader if provided (Native Bridge or direct-source bundle)
  const hasLoadShard = typeof runtimeStorageContext?.loadShard === 'function';
  const hasLoadShardRange = typeof runtimeStorageContext?.loadShardRange === 'function';
  const hasStreamShardRange = typeof runtimeStorageContext?.streamShardRange === 'function';
  if (hasLoadShard || hasLoadShardRange) {
    log.debug('Pipeline', 'Using custom shard loader (Native Bridge or external)');

    const loadShard = async (index) => {
      try {
        if (hasLoadShard) {
          const data = await runtimeStorageContext.loadShard(index);
          return toUint8Array(data, 'storageContext.loadShard');
        }
        const rangeData = await runtimeStorageContext.loadShardRange(index, 0, null);
        return toUint8Array(rangeData, 'storageContext.loadShardRange');
      } catch (error) {
        throw createStorageContextLoadError(error, hasLoadShard ? 'loadShard' : 'loadShardRange', index);
      }
    };

    const loadShardRange = hasLoadShardRange
      ? async (index, offset, length = null) => {
        try {
          const data = await runtimeStorageContext.loadShardRange(index, offset, length);
          return toArrayBuffer(data, 'storageContext.loadShardRange');
        } catch (error) {
          throw createStorageContextLoadError(error, 'loadShardRange', index, { offset, length });
        }
      }
      : null;

    const streamShardRange = hasStreamShardRange
      ? async function* (index, offset = 0, length = null, streamOptions = {}) {
        try {
          for await (const chunk of runtimeStorageContext.streamShardRange(index, offset, length, streamOptions)) {
            yield toUint8Array(chunk, 'storageContext.streamShardRange');
          }
        } catch (error) {
          throw createStorageContextLoadError(error, 'streamShardRange', index, { offset, length });
        }
      }
      : null;

    dopplerLoader.setCustomShardLoader(loadShard, {
      verify: verifyHashes,
      loadShardRange,
      rangeLease: runtimeStorageContext.rangeLease ?? null,
      loadShardRangeInto: typeof runtimeStorageContext.loadShardRangeInto === 'function'
        ? (index, offset, destination) => runtimeStorageContext.loadShardRangeInto(index, offset, destination)
        : null,
      streamShardRange,
      loadAuxiliaryFile: typeof runtimeStorageContext?.loadAuxiliaryFile === 'function'
        ? (path) => runtimeStorageContext.loadAuxiliaryFile(path)
        : null,
    });
    if (isRDRRManifest(manifest)) {
      dopplerLoader.setManifest(manifest);
    }
  }

  await dopplerLoader.init();

  // Load model via DopplerLoader
  const modelId = manifest.modelId;
  if (!modelId) {
    throw new Error('Manifest is missing modelId. Re-convert the model with modelId set.');
  }
  try {
    await dopplerLoader.load(modelId, {
      verifyHashes, partition: options.partition, signal: options.signal,
      onProgress: onProgress || ((info) => {
        // Shard and layer progress are logged by loader with source info
        if (info.stage !== 'layers' && info.stage !== 'shards') {
          log.verbose('Loader', `${info.stage}: ${Math.round(info.progress * 100)}%`);
        }
      }),
    });
  } catch (error) {
    throw rewriteWeightLoadError(error, { modelId });
  }

  const routedPolicy = typeof dopplerLoader.getRoutedExpertPolicy === 'function'
    ? dopplerLoader.getRoutedExpertPolicy() : loadingConfig?.expertCache?.qwen35Routed;
  if (routedPolicy?.enabled && routedPolicy.storageMode === 'tiered-compact') {
    try {
      if (typeof dopplerLoader.prepareRoutedExpertResidency !== 'function') {
        throw new Error('Tiered expert loading requires the residency preparation port.');
      }
      await dopplerLoader.prepareRoutedExpertResidency(options.signal);
    } catch (error) {
      try { await dopplerLoader.unload(); }
      catch (cleanup) { throw new AggregateError([error, cleanup], 'Tiered expert preload and loader retirement failed.'); }
      throw rewriteWeightLoadError(error, { modelId });
    }
  }

  // Map layer weights
  
  const layerWeights = new Map();
  for (let l = 0; l < modelConfig.numLayers; l++) {
    const weights = dopplerLoader.getLayerWeights(l);
    if (weights) {
      layerWeights.set(`layer_${l}`, weights);
    }
  }
  // Collect per-layer router weights for MoE
  
  const layerRouterWeights = new Map();
  if (modelConfig.useMoE) {
    for (let l = 0; l < modelConfig.numLayers; l++) {
      const weights = layerWeights.get(`layer_${l}`);
      if (weights?.routerWeight) {
        layerRouterWeights.set(l, {
          weight: weights.routerWeight,
          bias: weights.routerBias || null,
          scale: weights.routerScale || null,
          perExpertScale: weights.routerPerExpertScale || null,
        });
      }
    }
    log.debug('Pipeline', `MoE expert preparation: ${dopplerLoader.getLoadTiming()?.expertPreparation?.status}`);
  }

  return {
    loader: dopplerLoader,
    layerWeights,
    embeddings: dopplerLoader.embeddings,
    lmHead: dopplerLoader.lmHead,
    lmHeadBias: dopplerLoader.lmHeadBias,
    finalNorm: dopplerLoader.finalNorm,
    hyperConnectionMixer: dopplerLoader.hyperConnectionMixer,
    finalNormBias: dopplerLoader.finalNormBias,
    embeddingPostprocessor: dopplerLoader.embeddingPostprocessor,
    diffusionGemmaSelfConditioning: dopplerLoader.diffusionGemmaSelfConditioning,
    perLayerInputWeights: dopplerLoader.perLayerInputWeights,
    layerRouterWeights,
    loadTiming: typeof dopplerLoader.getLoadTiming === 'function'
      ? dopplerLoader.getLoadTiming()
      : null,
  };
}
