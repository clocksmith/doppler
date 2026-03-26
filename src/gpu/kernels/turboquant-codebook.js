
import { log } from '../../debug/index.js';

// =============================================================================
// TurboQuant Codebook & Rotation Matrix Module
//
// Precomputes Max-Lloyd scalar quantizer codebooks for the Beta distribution
// arising from random orthogonal rotation of unit-norm vectors, plus
// deterministic rotation matrix generation via Householder QR.
// =============================================================================

// -- Seeded PRNG (xoshiro128**) -----------------------------------------------

function createPRNG(seed) {
  const s = new Uint32Array(4);
  let z = seed | 0;
  for (let i = 0; i < 4; i++) {
    z = (z + 0x9e3779b9) | 0;
    let t = z ^ (z >>> 16);
    t = Math.imul(t, 0x85ebca6b);
    t = t ^ (t >>> 13);
    t = Math.imul(t, 0xc2b2ae35);
    s[i] = (t ^ (t >>> 16)) >>> 0;
  }

  function rotl(x, k) {
    return ((x << k) | (x >>> (32 - k))) >>> 0;
  }

  function nextU32() {
    const result = (Math.imul(rotl(Math.imul(s[1], 5), 7), 9)) >>> 0;
    const t = (s[1] << 9) >>> 0;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 11);
    return result;
  }

  function nextFloat() {
    return (nextU32() >>> 0) / 4294967296;
  }

  function nextGaussian() {
    const u1 = nextFloat() || 1e-10;
    const u2 = nextFloat();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  return { nextFloat, nextGaussian };
}

// -- Sphere marginal PDF ------------------------------------------------------
// The marginal distribution of a single coordinate of a uniform point on
// S^{d-1} is proportional to (1 - x^2)^((d-3)/2) on [-1, 1].

function sphereMarginalPDF(x, d) {
  if (d < 3) {
    throw new Error('TurboQuant requires headDim >= 3.');
  }
  const t = 1.0 - x * x;
  if (t <= 0) return 0;
  const exponent = (d - 3) / 2;
  return Math.pow(t, exponent);
}

// -- Numerical integration (Simpson's rule) -----------------------------------

function integrateWeighted(weightFn, valueFn, a, b, n) {
  if (n % 2 !== 0) n++;
  const h = (b - a) / n;
  let sumW = 0;
  let sumV = 0;
  for (let i = 0; i <= n; i++) {
    const x = a + i * h;
    const coeff = (i === 0 || i === n) ? 1 : (i % 2 === 0 ? 2 : 4);
    const w = weightFn(x) * coeff;
    sumW += w;
    sumV += w * valueFn(x);
  }
  return { integral: sumW * h / 3, moment: sumV * h / 3 };
}

function integrate(fn, a, b, n) {
  if (n % 2 !== 0) n++;
  const h = (b - a) / n;
  let sum = 0;
  for (let i = 0; i <= n; i++) {
    const x = a + i * h;
    const coeff = (i === 0 || i === n) ? 1 : (i % 2 === 0 ? 2 : 4);
    sum += fn(x) * coeff;
  }
  return sum * h / 3;
}

// -- Max-Lloyd quantizer for sphere marginal ----------------------------------

const QUADRATURE_POINTS = 2000;
const MAX_LLOYD_ITERATIONS = 200;
const LLOYD_CONVERGENCE_EPS = 1e-12;

/**
 * Compute the Max-Lloyd optimal scalar quantizer for the sphere marginal
 * distribution at a given dimension and bit-width.
 *
 * @param {number} d - Dimension (headDim).
 * @param {number} bitWidth - Bits per coordinate (1, 2, 3, or 4).
 * @returns {{ centroids: Float32Array, boundaries: Float32Array }}
 */
export function computeMaxLloydCodebook(d, bitWidth) {
  if (bitWidth < 1 || bitWidth > 4) {
    throw new Error(`TurboQuant bit-width must be 1-4; got ${bitWidth}.`);
  }
  const numCentroids = 1 << bitWidth;
  const numBoundaries = numCentroids - 1;
  const pdf = (x) => sphereMarginalPDF(x, d);

  // Initialize boundaries uniformly on [-1, 1]
  const boundaries = new Float64Array(numBoundaries);
  for (let i = 0; i < numBoundaries; i++) {
    boundaries[i] = -1.0 + (2.0 * (i + 1)) / numCentroids;
  }
  const centroids = new Float64Array(numCentroids);

  for (let iter = 0; iter < MAX_LLOYD_ITERATIONS; iter++) {
    // Update centroids: c_i = E[X | b_{i-1} <= X < b_i]
    let prevBound = -1.0;
    for (let i = 0; i < numCentroids; i++) {
      const nextBound = i < numBoundaries ? boundaries[i] : 1.0;
      const { integral: mass, moment } = integrateWeighted(
        pdf, (x) => x, prevBound, nextBound, QUADRATURE_POINTS
      );
      centroids[i] = mass > 0 ? moment / mass : (prevBound + nextBound) / 2;
      prevBound = nextBound;
    }

    // Update boundaries: b_i = (c_i + c_{i+1}) / 2
    let maxShift = 0;
    for (let i = 0; i < numBoundaries; i++) {
      const newBound = (centroids[i] + centroids[i + 1]) / 2;
      maxShift = Math.max(maxShift, Math.abs(newBound - boundaries[i]));
      boundaries[i] = newBound;
    }

    if (maxShift < LLOYD_CONVERGENCE_EPS) {
      break;
    }
  }

  return {
    centroids: new Float32Array(centroids),
    boundaries: new Float32Array(boundaries),
  };
}

// -- Codebook cache (per dimension × bit-width) -------------------------------

const codebookCache = new Map();

function codebookKey(d, bitWidth) {
  return `${d}:${bitWidth}`;
}

/**
 * Get or compute the Max-Lloyd codebook for a given dimension and bit-width.
 *
 * @param {number} d - Dimension (headDim).
 * @param {number} bitWidth - Bits per coordinate.
 * @returns {{ centroids: Float32Array, boundaries: Float32Array }}
 */
export function getCodebook(d, bitWidth) {
  const key = codebookKey(d, bitWidth);
  let cb = codebookCache.get(key);
  if (!cb) {
    cb = computeMaxLloydCodebook(d, bitWidth);
    codebookCache.set(key, cb);
    log.info('TurboQuant', `Computed codebook: d=${d}, b=${bitWidth}, centroids=${cb.centroids.length}`);
  }
  return cb;
}

// -- Rotation matrix generation -----------------------------------------------
// Generates a deterministic d×d orthogonal matrix via QR decomposition
// of a random Gaussian matrix using Householder reflections.

/**
 * Generate a deterministic d×d orthogonal rotation matrix.
 *
 * @param {number} d - Dimension (headDim).
 * @param {number} seed - PRNG seed for reproducibility.
 * @returns {Float32Array} - Flattened d×d orthogonal matrix (row-major).
 */
export function generateRotationMatrix(d, seed) {
  const rng = createPRNG(seed);

  // Generate random Gaussian matrix A (d × d)
  const A = new Float64Array(d * d);
  for (let i = 0; i < d * d; i++) {
    A[i] = rng.nextGaussian();
  }

  // Householder QR decomposition → extract Q
  const R = new Float64Array(A);
  const Q = new Float64Array(d * d);

  // Initialize Q as identity
  for (let i = 0; i < d; i++) {
    Q[i * d + i] = 1.0;
  }

  for (let k = 0; k < d; k++) {
    // Extract column k below diagonal
    const v = new Float64Array(d - k);
    for (let i = k; i < d; i++) {
      v[i - k] = R[i * d + k];
    }

    // Compute Householder vector
    let normV = 0;
    for (let i = 0; i < v.length; i++) {
      normV += v[i] * v[i];
    }
    normV = Math.sqrt(normV);

    if (normV < 1e-15) continue;

    const sign = v[0] >= 0 ? 1 : -1;
    v[0] += sign * normV;

    // Normalize v
    let normV2 = 0;
    for (let i = 0; i < v.length; i++) {
      normV2 += v[i] * v[i];
    }
    if (normV2 < 1e-30) continue;
    const invNorm = 2.0 / normV2;

    // Apply H = I - 2vv^T/||v||^2 to R columns k..d-1
    for (let j = k; j < d; j++) {
      let dot = 0;
      for (let i = 0; i < v.length; i++) {
        dot += v[i] * R[(i + k) * d + j];
      }
      dot *= invNorm;
      for (let i = 0; i < v.length; i++) {
        R[(i + k) * d + j] -= v[i] * dot;
      }
    }

    // Apply H to Q columns
    for (let j = 0; j < d; j++) {
      let dot = 0;
      for (let i = 0; i < v.length; i++) {
        dot += v[i] * Q[(i + k) * d + j];
      }
      dot *= invNorm;
      for (let i = 0; i < v.length; i++) {
        Q[(i + k) * d + j] -= v[i] * dot;
      }
    }
  }

  // Ensure det(Q) = +1 (proper rotation) by flipping first row if needed
  let det = 1.0;
  for (let i = 0; i < d; i++) {
    det *= R[i * d + i];
  }
  if (det < 0) {
    for (let j = 0; j < d; j++) {
      Q[j] = -Q[j]; // Flip first row
    }
  }

  return new Float32Array(Q);
}

// -- QJL projection matrix for TURBOQUANTprod --------------------------------
// The QJL (Quantized Johnson-Lindenstrauss) transform uses a random sign
// matrix for 1-bit residual quantization.

/**
 * Generate a deterministic d×d random sign matrix for QJL projection.
 * Each entry is +1 or -1 with equal probability, scaled by 1/sqrt(d).
 *
 * @param {number} d - Dimension (headDim).
 * @param {number} seed - PRNG seed (should differ from rotation seed).
 * @returns {Float32Array} - Flattened d×d sign matrix (row-major).
 */
export function generateQJLMatrix(d, seed) {
  const rng = createPRNG(seed);
  const P = new Float32Array(d * d);
  const scale = 1.0 / Math.sqrt(d);
  for (let i = 0; i < d * d; i++) {
    P[i] = rng.nextFloat() < 0.5 ? -scale : scale;
  }
  return P;
}

// -- Outlier fraction for non-integer bit-widths ------------------------------

/**
 * Compute the fraction of channels that should use higher precision
 * for a given effective bit-width.
 *
 * @param {number} effectiveBits - Target effective bits (e.g., 2.5, 3.5).
 * @param {number} bitsHigh - Bits for outlier channels.
 * @param {number} bitsLow - Bits for non-outlier channels.
 * @returns {number} - Fraction of channels using bitsHigh (0 to 1).
 */
export function computeOutlierFraction(effectiveBits, bitsHigh, bitsLow) {
  if (bitsHigh <= bitsLow) {
    throw new Error(`bitsHigh (${bitsHigh}) must be > bitsLow (${bitsLow}).`);
  }
  // effectiveBits = fraction * bitsHigh + (1 - fraction) * bitsLow
  const fraction = (effectiveBits - bitsLow) / (bitsHigh - bitsLow);
  if (fraction < 0 || fraction > 1) {
    throw new Error(
      `Effective bits ${effectiveBits} out of range [${bitsLow}, ${bitsHigh}].`
    );
  }
  return fraction;
}

/**
 * Resolve outlier bit-width config from effective bit-width.
 *
 * @param {number} effectiveBits - Target (e.g., 2.5, 3.5, or integer 2/3/4).
 * @returns {{ bitsHigh: number, bitsLow: number, outlierFraction: number, isNonInteger: boolean }}
 */
export function resolveOutlierConfig(effectiveBits) {
  if (Number.isInteger(effectiveBits) && effectiveBits >= 1 && effectiveBits <= 4) {
    return { bitsHigh: effectiveBits, bitsLow: effectiveBits, outlierFraction: 0, isNonInteger: false };
  }
  const bitsLow = Math.floor(effectiveBits);
  const bitsHigh = bitsLow + 1;
  if (bitsLow < 1 || bitsHigh > 4) {
    throw new Error(`Effective bits ${effectiveBits} out of supported range [1, 4].`);
  }
  const outlierFraction = computeOutlierFraction(effectiveBits, bitsHigh, bitsLow);
  return { bitsHigh, bitsLow, outlierFraction, isNonInteger: true };
}

// -- GPU buffer upload helpers ------------------------------------------------

/**
 * Upload rotation matrix to a GPU buffer.
 *
 * @param {GPUDevice} device - WebGPU device.
 * @param {Float32Array} matrix - Flattened d×d rotation matrix.
 * @param {string} label - Buffer label.
 * @returns {GPUBuffer}
 */
export function uploadRotationMatrix(device, matrix, label) {
  const buf = device.createBuffer({
    label,
    size: matrix.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, matrix);
  return buf;
}

/**
 * Upload codebook (centroids + boundaries) to GPU buffers.
 *
 * @param {GPUDevice} device - WebGPU device.
 * @param {{ centroids: Float32Array, boundaries: Float32Array }} codebook
 * @param {string} prefix - Buffer label prefix.
 * @returns {{ centroidsBuffer: GPUBuffer, boundariesBuffer: GPUBuffer }}
 */
export function uploadCodebook(device, codebook, prefix) {
  const centroidsBuffer = device.createBuffer({
    label: `${prefix}_centroids`,
    size: Math.max(4, codebook.centroids.byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(centroidsBuffer, 0, codebook.centroids);

  const boundariesBuffer = device.createBuffer({
    label: `${prefix}_boundaries`,
    size: Math.max(4, codebook.boundaries.byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(boundariesBuffer, 0, codebook.boundaries);

  return { centroidsBuffer, boundariesBuffer };
}

// -- TurboQuant packing helpers -----------------------------------------------

/**
 * Compute packed stride for a given headDim and bit-width.
 * packedStride = ceil(headDim / packFactor) where packFactor = floor(32 / bitWidth).
 *
 * @param {number} headDim
 * @param {number} bitWidth
 * @returns {number}
 */
export function computePackedStride(headDim, bitWidth) {
  const packFactor = Math.floor(32 / bitWidth);
  return Math.ceil(headDim / packFactor);
}

/**
 * Compute the pack factor for a given bit-width.
 *
 * @param {number} bitWidth
 * @returns {number}
 */
export function computePackFactor(bitWidth) {
  return Math.floor(32 / bitWidth);
}

// -- Default seeds ------------------------------------------------------------

/** Default seed for rotation matrix Π. */
export const ROTATION_SEED = 0x54515545; // "TQUE"

/** Default seed for QJL projection matrix P (must differ from rotation seed). */
export const QJL_SEED = 0x514A4C50; // "QJLP"
