import { getDevice } from '../../../gpu/device.js';
import { isBufferActive, releaseBuffer } from '../../../memory/buffer-pool.js';
import { isGpuBufferInstance } from '../../../gpu/weight-buffer.js';
import { log } from '../../../debug/index.js';
import { runRoPEPrecompute } from '../../../gpu/kernels/rope-precompute.js';
import { getShaderScopeCacheKey } from '../../../gpu/kernels/shader-source-scope.js';

// ============================================================================
// RoPE Initialization
// ============================================================================


async function computeRoPEFreqsForTheta(
  theta,
  rotaryDim,
  frequencyBaseDim,
  maxSeqLen,
  ropeScale,
  ropeScalingType,
  ropeScaling,
  inverseFrequencies
) {
  return runRoPEPrecompute({
    theta,
    rotaryDim,
    frequencyBaseDim,
    maxSeqLen,
    ropeScale,
    scalingType: ropeScalingType,
    scaling: ropeScaling,
    inverseFrequencies,
  });
}

function isSameRoPEScalingConfig(
  leftType,
  leftScale,
  leftScaling,
  rightType,
  rightScale,
  rightScaling
) {
  if (leftType !== rightType) return false;
  if (leftScale !== rightScale) return false;
  if (leftType === 'longrope') {
    return JSON.stringify(leftScaling?.short_factor ?? null) === JSON.stringify(rightScaling?.short_factor ?? null)
      && JSON.stringify(leftScaling?.long_factor ?? null) === JSON.stringify(rightScaling?.long_factor ?? null)
      && (leftScaling?.original_max_position_embeddings ?? null)
        === (rightScaling?.original_max_position_embeddings ?? null);
  }
  if (leftType !== 'yarn') return true;
  return (leftScaling?.beta_fast ?? null) === (rightScaling?.beta_fast ?? null)
    && (leftScaling?.beta_slow ?? null) === (rightScaling?.beta_slow ?? null)
    && (leftScaling?.original_max_position_embeddings ?? null)
      === (rightScaling?.original_max_position_embeddings ?? null);
}

const GPU_ROPE_BUFFER_CACHE = new WeakMap();
const GPU_ROPE_LEASES = new WeakMap();

function acquireRoPELease(entry) {
  const lease = { ...entry.buffers };
  entry.references += 1;
  GPU_ROPE_LEASES.set(lease, entry);
  return lease;
}

function releaseRoPEBuffers(buffers) {
  for (const buffer of new Set(Object.values(buffers))) {
    if (buffer && isBufferActive(buffer)) releaseBuffer(buffer);
  }
}

export function releaseRoPEFrequencies(lease) {
  const entry = lease && GPU_ROPE_LEASES.get(lease);
  if (!entry) return;
  GPU_ROPE_LEASES.delete(lease);
  entry.references -= 1;
  if (entry.references !== 0) return;
  if (entry.cache.get(entry.key) === entry) entry.cache.delete(entry.key);
  releaseRoPEBuffers(entry.buffers);
}
function isLiveCachedRopeBuffer(buffer) {
  return buffer == null || isBufferActive(buffer);
}

function hasLiveCachedGpuRopeBuffers(buffers) {
  return !!buffers
    && isLiveCachedRopeBuffer(buffers.cos)
    && isLiveCachedRopeBuffer(buffers.sin)
    && isLiveCachedRopeBuffer(buffers.localCos)
    && isLiveCachedRopeBuffer(buffers.localSin);
}

function buildRoPECacheKey(config) {
  return JSON.stringify({
    headDim: config.headDim,
    localHeadDim: config.localHeadDim ?? null,
    rotaryDim: config.rotaryDim ?? null,
    ropeLocalRotaryDim: config.ropeLocalRotaryDim ?? null,
    ropeInverseFrequencies: config.ropeInverseFrequencies ?? null,
    ropeFrequencyBaseDim: config.ropeFrequencyBaseDim ?? null,
    ropeLocalFrequencyBaseDim: config.ropeLocalFrequencyBaseDim ?? null,
    maxSeqLen: config.maxSeqLen,
    ropeTheta: config.ropeTheta,
    ropeLocalTheta: config.ropeLocalTheta ?? null,
    mropeInterleaved: config.mropeInterleaved === true,
    mropeSection: Array.isArray(config.mropeSection) ? [...config.mropeSection] : null,
    partialRotaryFactor: config.partialRotaryFactor ?? null,
    ropeLocalPartialRotaryFactor: config.ropeLocalPartialRotaryFactor ?? null,
    ropeScale: config.ropeScale,
    ropeLocalScale: config.ropeLocalScale ?? null,
    ropeScalingType: config.ropeScalingType ?? null,
    ropeLocalScalingType: config.ropeLocalScalingType ?? null,
    ropeScaling: config.ropeScaling ?? null,
    ropeLocalScaling: config.ropeLocalScaling ?? null,
  });
}

function resolveRotaryDim(headDim, rotaryDim, partialRotaryFactor) {
  if (rotaryDim != null) {
    if (!Number.isFinite(rotaryDim) || rotaryDim <= 0 || (rotaryDim % 2) !== 0) {
      throw new Error(`RoPE rotary dim must be a positive even integer; got "${rotaryDim}".`);
    }
    if (rotaryDim > headDim) {
      throw new Error(`RoPE rotary dim ${rotaryDim} cannot exceed headDim ${headDim}.`);
    }
    return rotaryDim;
  }
  if (partialRotaryFactor == null) {
    return headDim;
  }
  if (!Number.isFinite(partialRotaryFactor) || partialRotaryFactor <= 0 || partialRotaryFactor > 1) {
    throw new Error(
      `RoPE partialRotaryFactor must be a number in (0, 1]; got "${partialRotaryFactor}".`
    );
  }
  const resolved = Math.trunc(headDim * partialRotaryFactor);
  if (resolved <= 0 || (resolved % 2) !== 0) {
    throw new Error(
      `RoPE partialRotaryFactor=${partialRotaryFactor} with headDim=${headDim} resolves ` +
      `to rotaryDim=${resolved}, but rotaryDim must be a positive even integer.`
    );
  }
  return resolved;
}

function resolveFrequencyBaseDim(rotaryDim, frequencyBaseDim, label) {
  if (frequencyBaseDim == null) {
    return rotaryDim;
  }
  if (!Number.isFinite(frequencyBaseDim) || frequencyBaseDim <= 0 || (Math.trunc(frequencyBaseDim) % 2) !== 0) {
    throw new Error(`${label} must be a positive even integer; got "${frequencyBaseDim}".`);
  }
  const resolved = Math.trunc(frequencyBaseDim);
  if (resolved < rotaryDim) {
    throw new Error(`${label} ${resolved} cannot be smaller than rotaryDim ${rotaryDim}.`);
  }
  return resolved;
}


export async function initRoPEFrequencies(config, useGPU) {
  const cacheKey = `${getShaderScopeCacheKey()}:${buildRoPECacheKey(config)}`;
  const {
    headDim,
    localHeadDim,
    rotaryDim,
    ropeLocalRotaryDim,
    ropeInverseFrequencies,
    ropeFrequencyBaseDim,
    ropeLocalFrequencyBaseDim,
    maxSeqLen,
    ropeTheta,
    ropeLocalTheta,
    mropeInterleaved,
    mropeSection,
    partialRotaryFactor,
    ropeLocalPartialRotaryFactor,
    ropeScale,
    ropeLocalScale,
    ropeScalingType,
    ropeLocalScalingType,
    ropeScaling,
    ropeLocalScaling,
  } = config;
  if (!Number.isFinite(ropeScale) || ropeScale <= 0) {
    throw new Error(`RoPE scale must be a positive number; got "${ropeScale}".`);
  }
  const resolvedLocalScale = ropeLocalScale;
  if (resolvedLocalScale != null && (!Number.isFinite(resolvedLocalScale) || resolvedLocalScale <= 0)) {
    throw new Error(`Local RoPE scale must be a positive number; got "${resolvedLocalScale}".`);
  }
  const resolvedLocalTheta = ropeLocalTheta ?? ropeTheta;
  const resolvedLocalScalingType = (
    ropeLocalScalingType === undefined
      ? ropeScalingType
      : ropeLocalScalingType
  );
  const resolvedLocalScaling = (
    ropeLocalScalingType === undefined
      ? ropeScaling
      : ropeLocalScaling
  );
  const resolvedLocalHeadDim = localHeadDim ?? headDim;
  const resolvedRotaryDim = resolveRotaryDim(headDim, rotaryDim, partialRotaryFactor);
  const resolvedLocalRotaryDim = resolveRotaryDim(
    resolvedLocalHeadDim,
    ropeLocalRotaryDim,
    ropeLocalPartialRotaryFactor
  );
  const resolvedFrequencyBaseDim = resolveFrequencyBaseDim(
    resolvedRotaryDim,
    ropeFrequencyBaseDim,
    'RoPE frequency base dim'
  );
  const resolvedLocalFrequencyBaseDim = resolveFrequencyBaseDim(
    resolvedLocalRotaryDim,
    ropeLocalFrequencyBaseDim,
    'Local RoPE frequency base dim'
  );
  const halfDim = resolvedRotaryDim / 2;
  if (mropeInterleaved === true && Array.isArray(mropeSection)) {
    const expandedDim = mropeSection.reduce((sum, entry) => sum + entry, 0) * 2;
    if (expandedDim !== resolvedRotaryDim) {
      throw new Error(
        `RoPE mropeSection expands to ${expandedDim} dims, but rotaryDim is ${resolvedRotaryDim}.`
      );
    }
  }

  const isYarn = ropeScalingType === 'yarn';
  const isLocalYarn = resolvedLocalScalingType === 'yarn';

  const device = getDevice();
  if (!useGPU || !device) {
    throw new Error('RoPE frequency initialization requires the declared WebGPU execution path.');
  }
  let perDeviceCache = GPU_ROPE_BUFFER_CACHE.get(device);
  if (!perDeviceCache) {
    perDeviceCache = new Map();
    GPU_ROPE_BUFFER_CACHE.set(device, perDeviceCache);
  }
  const cachedEntry = perDeviceCache.get(cacheKey);
  if (cachedEntry) {
    if (hasLiveCachedGpuRopeBuffers(cachedEntry.buffers)) {
      return acquireRoPELease(cachedEntry);
    }
    perDeviceCache.delete(cacheKey);
  }

  // Compute global (full_attention) frequencies
  const globalFreqs = await computeRoPEFreqsForTheta(
    ropeTheta,
    resolvedRotaryDim,
    resolvedFrequencyBaseDim,
    maxSeqLen,
    ropeScale,
    ropeScalingType,
    ropeScaling,
    ropeInverseFrequencies
  );

  // Compute local (sliding_attention) frequencies if different from global.
  // Models with dual RoPE use different theta for local vs global attention layers.

  let localFreqs = null;
  const hasDistinctLocalTheta = resolvedLocalTheta !== ropeTheta;
  const hasDistinctLocalDim = resolvedLocalRotaryDim !== resolvedRotaryDim;
  const hasDistinctLocalScaling = !isSameRoPEScalingConfig(
    ropeScalingType,
    ropeScale,
    ropeScaling,
    resolvedLocalScalingType,
    resolvedLocalScale,
    resolvedLocalScaling
  );
  if (hasDistinctLocalTheta || hasDistinctLocalScaling || hasDistinctLocalDim) {
    try {
      localFreqs = await computeRoPEFreqsForTheta(
      resolvedLocalTheta,
      resolvedLocalRotaryDim,
      resolvedLocalFrequencyBaseDim,
      maxSeqLen,
      resolvedLocalScale,
      resolvedLocalScalingType,
      resolvedLocalScaling,
      null
      );
    } catch (error) {
      releaseRoPEBuffers(globalFreqs);
      throw error;
    }
    log.debug(
      'Pipeline',
      `Dual RoPE: local theta=${resolvedLocalTheta}, global theta=${ropeTheta}, ` +
      `localRotaryDim=${resolvedLocalRotaryDim}, globalRotaryDim=${resolvedRotaryDim}, ` +
      `localFrequencyBaseDim=${resolvedLocalFrequencyBaseDim}, globalFrequencyBaseDim=${resolvedFrequencyBaseDim}, ` +
      `localScaling=${resolvedLocalScalingType ?? 'none'}:${resolvedLocalScale}, ` +
      `globalScaling=${ropeScalingType ?? 'none'}:${ropeScale}`
    );
  }

  if (isYarn) {
    // Log YARN params (already validated in computeRoPEFreqs)
    log.debug('Pipeline', `YARN RoPE: factor=${ropeScaling?.factor}, beta_fast=${ropeScaling?.beta_fast}, beta_slow=${ropeScaling?.beta_slow}`);
  }
  if (isLocalYarn && hasDistinctLocalScaling) {
    log.debug(
      'Pipeline',
      `Local YARN RoPE: factor=${resolvedLocalScaling?.factor}, ` +
      `beta_fast=${resolvedLocalScaling?.beta_fast}, beta_slow=${resolvedLocalScaling?.beta_slow}`
    );
  }

  log.debug(
    'Pipeline',
    `RoPE frequencies initialized (GPU): ${maxSeqLen} positions, dim=${halfDim}, headDim=${headDim}, rotaryDim=${resolvedRotaryDim}, ` +
    `theta=${ropeTheta}${hasDistinctLocalTheta ? `, localTheta=${resolvedLocalTheta}` : ''}, ` +
    `${hasDistinctLocalDim ? `localRotaryDim=${resolvedLocalRotaryDim}, ` : ''}` +
    `scaling=${ropeScalingType ?? 'none'}:${ropeScale}${hasDistinctLocalScaling ? `, localScaling=${resolvedLocalScalingType ?? 'none'}:${resolvedLocalScale}` : ''}, ` +
    `interleaved=${mropeInterleaved === true}`
  );

  const buffers = {
    cos: globalFreqs.cos,
    sin: globalFreqs.sin,
    localCos: localFreqs?.cos ?? null,
    localSin: localFreqs?.sin ?? null,
  };
  const entry = { buffers, references: 0, cache: perDeviceCache, key: cacheKey };
  perDeviceCache.set(cacheKey, entry);
  return acquireRoPELease(entry);
}


export function isGPURoPEBuffers(buffers) {
  if (typeof GPUBuffer === 'undefined') return false;
  return !!buffers?.cos && isGpuBufferInstance(buffers.cos);
}

export async function _initRoPE() {
    const config = (this.modelConfig);
    const maxSeqLen = config.maxSeqLen;
    const ropeBuffers = await initRoPEFrequencies({
      headDim: config.globalHeadDim ?? config.headDim,
      localHeadDim: config.headDim,
      rotaryDim: config.ropeRotaryDim,
      ropeLocalRotaryDim: config.ropeLocalRotaryDim,
      ropeInverseFrequencies: config.ropeInverseFrequencies,
      ropeFrequencyBaseDim: config.ropeFrequencyBaseDim,
      ropeLocalFrequencyBaseDim: config.ropeLocalFrequencyBaseDim,
      maxSeqLen,
      ropeTheta: config.ropeTheta,
      ropeLocalTheta: config.ropeLocalTheta,
      mropeInterleaved: config.mropeInterleaved,
      mropeSection: config.mropeSection,
      partialRotaryFactor: config.partialRotaryFactor,
      ropeLocalPartialRotaryFactor: config.ropeLocalPartialRotaryFactor,
      ropeScale: config.ropeScale,
      ropeLocalScale: config.ropeLocalScale,
      ropeScalingType: config.ropeScalingType,
      ropeLocalScalingType: config.ropeLocalScalingType,
      ropeScaling: config.ropeScaling,
      ropeLocalScaling: config.ropeLocalScaling,
    }, this.useGPU);
    releaseRoPEFrequencies(this.ropeFrequencyLease);
    this.ropeFrequencyLease = ropeBuffers;
    this.ropeFreqsCos = ropeBuffers.cos;
    this.ropeFreqsSin = ropeBuffers.sin;
    this.ropeLocalCos = ropeBuffers.localCos ?? null;
    this.ropeLocalSin = ropeBuffers.localSin ?? null;
  }
