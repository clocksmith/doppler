function toUint8ArrayView(value, label) {
  if (value instanceof Uint8ClampedArray) {
    return value;
  }
  if (value instanceof Uint8Array) {
    return value;
  }
  throw new Error(`${label} must be a Uint8Array or Uint8ClampedArray.`);
}

function toToleranceValue(value, fallback) {
  return Number.isFinite(value) ? Number(value) : fallback;
}

export function computeImageFingerprint(pixels) {
  const view = toUint8ArrayView(pixels, 'pixels');
  let hash = 0x811c9dc5;
  for (let i = 0; i < view.length; i += 1) {
    hash ^= view[i];
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, '0');
}

export function computeImageRegressionMetrics(actualPixels, expectedPixels) {
  const actual = toUint8ArrayView(actualPixels, 'actualPixels');
  const expected = toUint8ArrayView(expectedPixels, 'expectedPixels');
  if (actual.length !== expected.length) {
    throw new Error(
      `Image regression requires equal-length buffers: actual=${actual.length}, expected=${expected.length}.`
    );
  }

  const count = actual.length;
  if (count === 0) {
    return {
      samples: 0,
      mae: 0,
      mse: 0,
      rmse: 0,
      maxAbsDiff: 0,
      psnr: Infinity,
    };
  }

  let sumAbs = 0;
  let sumSq = 0;
  let maxAbsDiff = 0;
  for (let i = 0; i < count; i += 1) {
    const diff = Math.abs(actual[i] - expected[i]);
    sumAbs += diff;
    sumSq += diff * diff;
    if (diff > maxAbsDiff) {
      maxAbsDiff = diff;
    }
  }

  const mae = sumAbs / count;
  const mse = sumSq / count;
  const rmse = Math.sqrt(mse);
  const psnr = rmse === 0 ? Infinity : (20 * Math.log10(255 / rmse));

  return {
    samples: count,
    mae,
    mse,
    rmse,
    maxAbsDiff,
    psnr,
  };
}

export function assertImageRegressionWithinTolerance(
  actualPixels,
  expectedPixels,
  tolerance = {},
  contextLabel = 'image-regression'
) {
  const metrics = computeImageRegressionMetrics(actualPixels, expectedPixels);
  const maxAbsDiffLimit = toToleranceValue(tolerance.maxAbsDiff, 0);
  const maeLimit = toToleranceValue(tolerance.mae, 0);
  const rmseLimit = toToleranceValue(tolerance.rmse, 0);
  const minPsnr = toToleranceValue(tolerance.minPsnr, Infinity);

  if (metrics.maxAbsDiff > maxAbsDiffLimit) {
    throw new Error(
      `${contextLabel}: maxAbsDiff=${metrics.maxAbsDiff.toFixed(6)} exceeded limit=${maxAbsDiffLimit}.`
    );
  }
  if (metrics.mae > maeLimit) {
    throw new Error(
      `${contextLabel}: mae=${metrics.mae.toFixed(6)} exceeded limit=${maeLimit}.`
    );
  }
  if (metrics.rmse > rmseLimit) {
    throw new Error(
      `${contextLabel}: rmse=${metrics.rmse.toFixed(6)} exceeded limit=${rmseLimit}.`
    );
  }
  if (Number.isFinite(minPsnr) && metrics.psnr < minPsnr) {
    throw new Error(
      `${contextLabel}: psnr=${metrics.psnr.toFixed(6)} was below minimum=${minPsnr}.`
    );
  }

  return metrics;
}
