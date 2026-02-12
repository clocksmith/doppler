


export const KERNEL_TOLERANCES = {
  matmul_f32: { rtol: 1e-5, atol: 1e-6 },
  matmul_f16: { rtol: 1e-2, atol: 1e-3 }, // FP16 has ~3 decimal digits

  attention: { rtol: 1e-4, atol: 1e-5 }, // Softmax accumulation

  softmax: { rtol: 1e-5, atol: 1e-7 }, // Must sum to 1

  rmsnorm: { rtol: 1e-5, atol: 1e-6 },

  rope: { rtol: 1e-5, atol: 1e-6 }, // Sin/cos operations

  silu: { rtol: 1e-5, atol: 1e-6 },

  topk: {
    indices: { exact: true }, // Indices must match exactly
    weights: { rtol: 1e-5, atol: 1e-7 },
  },

  scatter_add: { rtol: 1e-5, atol: 1e-6 },

  moe_gather: { rtol: 1e-5, atol: 1e-6 },

  gather: { exact: true }, // Embedding lookup is exact

  residual: { rtol: 1e-6, atol: 1e-8 }, // Simple addition

  dequant: { rtol: 1e-4, atol: 1e-5 }, // Quantization introduces error
};


export function compareArrays(expected, actual, options = {}) {
  const { rtol = 1e-5, atol = 1e-8 } = options;

  if (expected.length !== actual.length) {
    return {
      passed: false,
      error: `Length mismatch: expected ${expected.length}, got ${actual.length}`,
      maxError: Infinity,
      avgError: Infinity,
      mismatchCount: expected.length,
    };
  }

  let maxError = 0;
  let sumError = 0;
  let mismatchCount = 0;
  const mismatches = [];

  for (let i = 0; i < expected.length; i++) {
    const e = expected[i];
    const a = actual[i];
    const error = Math.abs(e - a);
    const threshold = atol + rtol * Math.abs(e);

    maxError = Math.max(maxError, error);
    sumError += error;

    if (error > threshold) {
      mismatchCount++;
      if (mismatches.length < 10) {
        mismatches.push({ index: i, expected: e, actual: a, error, threshold });
      }
    }
  }

  return {
    passed: mismatchCount === 0,
    maxError,
    avgError: sumError / expected.length,
    mismatchCount,
    mismatchRatio: mismatchCount / expected.length,
    firstMismatches: mismatches,
  };
}


export function compareIntArrays(expected, actual) {
  if (expected.length !== actual.length) {
    return {
      passed: false,
      error: `Length mismatch: expected ${expected.length}, got ${actual.length}`,
      mismatchCount: expected.length,
    };
  }

  let mismatchCount = 0;
  const mismatches = [];

  for (let i = 0; i < expected.length; i++) {
    if (expected[i] !== actual[i]) {
      mismatchCount++;
      if (mismatches.length < 10) {
        mismatches.push({ index: i, expected: expected[i], actual: actual[i] });
      }
    }
  }

  return {
    passed: mismatchCount === 0,
    mismatchCount,
    firstMismatches: mismatches,
  };
}


export function generateTestData(size, seed = 42, options = {}) {
  const { min = -1, max = 1, dtype = 'float32' } = options;

  let data;
  switch (dtype) {
    case 'uint32':
      data = new Uint32Array(size);
      break;
    case 'int32':
      data = new Int32Array(size);
      break;
    default:
      data = new Float32Array(size);
  }

  // Simple LCG PRNG for reproducibility
  let state = seed;
  const range = max - min;

  for (let i = 0; i < size; i++) {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    const normalized = state / 0x7fffffff; // [0, 1]

    if (dtype === 'float32') {
      data[i] = min + normalized * range;
    } else {
      data[i] = Math.floor(min + normalized * range);
    }
  }

  return data;
}


export function verifySumTo(arr, expectedSum, tolerance = 1e-5) {
  const actualSum = arr.reduce((a, b) => a + b, 0);
  const error = Math.abs(actualSum - expectedSum);

  return {
    passed: error < tolerance,
    expectedSum,
    actualSum,
    error,
  };
}


export function verifyRange(arr, min, max) {
  let outOfRange = 0;
  let minVal = Infinity;
  let maxVal = -Infinity;

  for (let i = 0; i < arr.length; i++) {
    if (arr[i] < min || arr[i] > max) outOfRange++;
    minVal = Math.min(minVal, arr[i]);
    maxVal = Math.max(maxVal, arr[i]);
  }

  return {
    passed: outOfRange === 0,
    outOfRangeCount: outOfRange,
    actualMin: minVal,
    actualMax: maxVal,
  };
}
