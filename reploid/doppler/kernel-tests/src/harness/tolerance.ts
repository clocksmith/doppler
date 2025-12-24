/**
 * Floating-Point Tolerance and Comparison Utilities
 */

export interface ToleranceConfig {
  rtol: number;
  atol: number;
}

export interface TopKToleranceConfig {
  indices: { exact: boolean };
  weights: ToleranceConfig;
}

export interface KernelTolerances {
  matmul_f32: ToleranceConfig;
  matmul_f16: ToleranceConfig;
  attention: ToleranceConfig;
  softmax: ToleranceConfig;
  rmsnorm: ToleranceConfig;
  rope: ToleranceConfig;
  silu: ToleranceConfig;
  topk: TopKToleranceConfig;
  scatter_add: ToleranceConfig;
  moe_gather: ToleranceConfig;
  gather: { exact: boolean };
  residual: ToleranceConfig;
  dequant: ToleranceConfig;
}

/**
 * Per-kernel tolerance settings based on numerical characteristics
 */
export const KERNEL_TOLERANCES: KernelTolerances = {
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

export interface Mismatch {
  index: number;
  expected: number;
  actual: number;
  error: number;
  threshold: number;
}

export interface ComparisonResult {
  passed: boolean;
  error?: string;
  maxError: number;
  avgError: number;
  mismatchCount: number;
  mismatchRatio?: number;
  firstMismatches?: Mismatch[];
}

/**
 * Compare floating-point arrays with configurable tolerance
 * @param expected Expected values
 * @param actual Actual values
 * @param options Tolerance configuration
 * @returns Comparison results
 */
export function compareArrays(
  expected: Float32Array,
  actual: Float32Array,
  options: Partial<ToleranceConfig> = {}
): ComparisonResult {
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
  const mismatches: Mismatch[] = [];

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

export interface IntMismatch {
  index: number;
  expected: number;
  actual: number;
}

export interface IntComparisonResult {
  passed: boolean;
  error?: string;
  mismatchCount: number;
  firstMismatches?: IntMismatch[];
}

/**
 * Compare integer arrays (exact match)
 * @param expected Expected values
 * @param actual Actual values
 * @returns Comparison results
 */
export function compareIntArrays(
  expected: Uint32Array | Int32Array,
  actual: Uint32Array | Int32Array
): IntComparisonResult {
  if (expected.length !== actual.length) {
    return {
      passed: false,
      error: `Length mismatch: expected ${expected.length}, got ${actual.length}`,
      mismatchCount: expected.length,
    };
  }

  let mismatchCount = 0;
  const mismatches: IntMismatch[] = [];

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

export type DataType = 'float32' | 'int32' | 'uint32';

export interface GenerateOptions {
  min?: number;
  max?: number;
  dtype?: DataType;
}

/**
 * Generate deterministic test data using LCG PRNG
 * @param size Number of elements
 * @param seed Random seed
 * @param options Generation options
 * @returns Generated array
 */
export function generateTestData(
  size: number,
  seed: number = 42,
  options: GenerateOptions = {}
): Float32Array | Int32Array | Uint32Array {
  const { min = -1, max = 1, dtype = 'float32' } = options;

  let data: Float32Array | Int32Array | Uint32Array;
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

export interface SumVerificationResult {
  passed: boolean;
  expectedSum: number;
  actualSum: number;
  error: number;
}

/**
 * Verify array sums to expected value (e.g., softmax sums to 1)
 * @param arr Input array
 * @param expectedSum Expected sum
 * @param tolerance Error tolerance
 * @returns Verification result
 */
export function verifySumTo(
  arr: Float32Array,
  expectedSum: number,
  tolerance: number = 1e-5
): SumVerificationResult {
  const actualSum = arr.reduce((a, b) => a + b, 0);
  const error = Math.abs(actualSum - expectedSum);

  return {
    passed: error < tolerance,
    expectedSum,
    actualSum,
    error,
  };
}

export interface RangeVerificationResult {
  passed: boolean;
  outOfRangeCount: number;
  actualMin: number;
  actualMax: number;
}

/**
 * Verify all elements are in range
 * @param arr Input array
 * @param min Minimum value
 * @param max Maximum value
 * @returns Verification result
 */
export function verifyRange(arr: Float32Array, min: number, max: number): RangeVerificationResult {
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
