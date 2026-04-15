import { getDevice } from '../../../gpu/device.js';
import { acquireBuffer, isBufferActive, releaseBuffer } from '../../../memory/buffer-pool.js';
import { isGpuBufferInstance } from '../../../gpu/weight-buffer.js';
import { log } from '../../../debug/index.js';

// ============================================================================
// RoPE Initialization
// ============================================================================


function computeRoPEFreqsForTheta(
  theta,
  rotaryDim,
  frequencyBaseDim,
  maxSeqLen,
  ropeScale,
  ropeScalingType,
  ropeScaling
) {
  const halfDim = rotaryDim / 2;

  // Compute base frequencies: theta_i = 1 / (base^(2i/d))
  const freqs = new Float32Array(halfDim);
  for (let i = 0; i < halfDim; i++) {
    freqs[i] = 1.0 / Math.pow(theta, (2 * i) / frequencyBaseDim);
  }

  // Compute per-dimension scaling factors
  const scales = new Float32Array(halfDim);
  const isYarn = ropeScalingType === 'yarn';
  if (isYarn) {
    // YARN scaling - validate ALL required params (fail fast on incomplete manifest)
    if (ropeScaling?.factor == null ||
        ropeScaling?.beta_fast == null || ropeScaling?.beta_slow == null ||
        ropeScaling?.original_max_position_embeddings == null) {
      throw new Error(
        `RoPE scaling type is 'yarn' but YARN params missing. ` +
        `Manifest must provide factor, beta_fast, beta_slow, and original_max_position_embeddings. ` +
        `Got: factor=${ropeScaling?.factor}, beta_fast=${ropeScaling?.beta_fast}, beta_slow=${ropeScaling?.beta_slow}, ` +
        `original_max_position_embeddings=${ropeScaling?.original_max_position_embeddings}`
      );
    }
    // Extract validated YARN params (no hidden defaults - all guaranteed non-null)
    const yarnFactor = ropeScaling.factor;
    const yarnBetaFast = ropeScaling.beta_fast;
    const yarnBetaSlow = ropeScaling.beta_slow;
    const originalMaxPos = ropeScaling.original_max_position_embeddings;

    // YARN: wavelength-based interpolation
    for (let i = 0; i < halfDim; i++) {
      const wavelength = (2 * Math.PI) / freqs[i];
      const lowThresh = originalMaxPos / yarnBetaSlow;
      const highThresh = originalMaxPos / yarnBetaFast;

      if (wavelength < highThresh) {
        scales[i] = 1.0;
      } else if (wavelength > lowThresh) {
        scales[i] = yarnFactor;
      } else {
        const t = (wavelength - highThresh) / (lowThresh - highThresh);
        scales[i] = 1.0 + (yarnFactor - 1.0) * t;
      }
    }
  } else {
    // Linear scaling: uniform across all dimensions
    if (ropeScalingType != null && ropeScalingType !== 'linear') {
      log.warn(
        'Pipeline',
        `Unrecognized RoPE scaling type "${ropeScalingType}"; falling back to linear scaling. ` +
        'Known types: "linear", "yarn".'
      );
    }
    for (let i = 0; i < halfDim; i++) {
      scales[i] = ropeScale;
    }
  }

  // Compute cos/sin for each position
  const cosValues = new Float32Array(maxSeqLen * halfDim);
  const sinValues = new Float32Array(maxSeqLen * halfDim);

  for (let pos = 0; pos < maxSeqLen; pos++) {
    for (let i = 0; i < halfDim; i++) {
      const scaledPos = pos / scales[i];
      const angle = scaledPos * freqs[i];
      cosValues[pos * halfDim + i] = Math.cos(angle);
      sinValues[pos * halfDim + i] = Math.sin(angle);
    }
  }

  return { cos: cosValues, sin: sinValues };
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
  if (leftType !== 'yarn') return true;
  return (leftScaling?.beta_fast ?? null) === (rightScaling?.beta_fast ?? null)
    && (leftScaling?.beta_slow ?? null) === (rightScaling?.beta_slow ?? null)
    && (leftScaling?.original_max_position_embeddings ?? null)
      === (rightScaling?.original_max_position_embeddings ?? null);
}

const GPU_ROPE_BUFFER_CACHE = new WeakMap();
const CPU_ROPE_BUFFER_CACHE = new Map();

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
  const cacheKey = buildRoPECacheKey(config);
  const {
    headDim,
    localHeadDim,
    rotaryDim,
    ropeLocalRotaryDim,
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

  // Compute global (full_attention) frequencies
  const globalFreqs = computeRoPEFreqsForTheta(
    ropeTheta,
    resolvedRotaryDim,
    resolvedFrequencyBaseDim,
    maxSeqLen,
    ropeScale,
    ropeScalingType,
    ropeScaling
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
    localFreqs = computeRoPEFreqsForTheta(
      resolvedLocalTheta,
      resolvedLocalRotaryDim,
      resolvedLocalFrequencyBaseDim,
      maxSeqLen,
      resolvedLocalScale,
      resolvedLocalScalingType,
      resolvedLocalScaling
    );
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

  // Upload to GPU if available
  const device = getDevice();
  if (device && useGPU) {
    let perDeviceCache = GPU_ROPE_BUFFER_CACHE.get(device);
    if (!perDeviceCache) {
      perDeviceCache = new Map();
      GPU_ROPE_BUFFER_CACHE.set(device, perDeviceCache);
    }
    const cachedBuffers = perDeviceCache.get(cacheKey);
    if (cachedBuffers) {
      if (hasLiveCachedGpuRopeBuffers(cachedBuffers)) {
        return cachedBuffers;
      }
      perDeviceCache.delete(cacheKey);
    }
    let cosBuffer = null;
    let sinBuffer = null;
    let localCosBuffer = null;
    let localSinBuffer = null;
    try {
      cosBuffer = acquireBuffer(globalFreqs.cos.byteLength, undefined, 'rope_cos');
      sinBuffer = acquireBuffer(globalFreqs.sin.byteLength, undefined, 'rope_sin');
      device.queue.writeBuffer(cosBuffer, 0, globalFreqs.cos.buffer, globalFreqs.cos.byteOffset, globalFreqs.cos.byteLength);
      device.queue.writeBuffer(sinBuffer, 0, globalFreqs.sin.buffer, globalFreqs.sin.byteOffset, globalFreqs.sin.byteLength);

      if (localFreqs) {
        localCosBuffer = acquireBuffer(localFreqs.cos.byteLength, undefined, 'rope_local_cos');
        localSinBuffer = acquireBuffer(localFreqs.sin.byteLength, undefined, 'rope_local_sin');
        device.queue.writeBuffer(localCosBuffer, 0, localFreqs.cos.buffer, localFreqs.cos.byteOffset, localFreqs.cos.byteLength);
        device.queue.writeBuffer(localSinBuffer, 0, localFreqs.sin.buffer, localFreqs.sin.byteOffset, localFreqs.sin.byteLength);
      }
    } catch (error) {
      for (const buffer of [cosBuffer, sinBuffer, localCosBuffer, localSinBuffer]) {
        if (buffer) {
          releaseBuffer(buffer);
        }
      }
      throw error;
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
      cos: cosBuffer,
      sin: sinBuffer,
      localCos: localCosBuffer,
      localSin: localSinBuffer,
    };
    perDeviceCache.set(cacheKey, buffers);
    return buffers;
  }

  const cachedCpuBuffers = CPU_ROPE_BUFFER_CACHE.get(cacheKey);
  if (cachedCpuBuffers) {
    return cachedCpuBuffers;
  }

  log.debug(
    'Pipeline',
    `RoPE frequencies initialized (CPU): ${maxSeqLen} positions, dim=${halfDim}, headDim=${headDim}, rotaryDim=${resolvedRotaryDim}, ` +
    `theta=${ropeTheta}${hasDistinctLocalTheta ? `, localTheta=${resolvedLocalTheta}` : ''}, ` +
    `${hasDistinctLocalDim ? `localRotaryDim=${resolvedLocalRotaryDim}, ` : ''}` +
    `scaling=${ropeScalingType ?? 'none'}:${ropeScale}${hasDistinctLocalScaling ? `, localScaling=${resolvedLocalScalingType ?? 'none'}:${resolvedLocalScale}` : ''}, ` +
    `interleaved=${mropeInterleaved === true}`
  );

  const buffers = {
    cos: globalFreqs.cos,
    sin: globalFreqs.sin,
    localCos: localFreqs?.cos,
    localSin: localFreqs?.sin,
  };
  CPU_ROPE_BUFFER_CACHE.set(cacheKey, buffers);
  return buffers;
}


export function isGPURoPEBuffers(buffers) {
  if (typeof GPUBuffer === 'undefined') return false;
  return !!buffers?.cos && isGpuBufferInstance(buffers.cos);
}
