/**
 * Reference Residual Add Implementation
 */

/**
 * Reference residual add
 * output = x + residual
 *
 * @param x Input
 * @param residual Residual connection
 * @returns Sum
 */
export function residualAddRef(x: Float32Array, residual: Float32Array): Float32Array {
  const output = new Float32Array(x.length);

  for (let i = 0; i < x.length; i++) {
    output[i] = x[i] + residual[i];
  }

  return output;
}

/**
 * In-place residual add (modifies x)
 */
export function residualAddInplaceRef(x: Float32Array, residual: Float32Array): Float32Array {
  for (let i = 0; i < x.length; i++) {
    x[i] += residual[i];
  }
  return x;
}

/**
 * Scaled residual add
 * output = x + scale * residual
 */
export function scaledResidualAddRef(x: Float32Array, residual: Float32Array, scale: number): Float32Array {
  const output = new Float32Array(x.length);

  for (let i = 0; i < x.length; i++) {
    output[i] = x[i] + scale * residual[i];
  }

  return output;
}

/**
 * Residual add with dropout mask
 * output = x + residual * mask * (1 / (1 - dropProb))
 */
export function residualAddDropoutRef(
  x: Float32Array,
  residual: Float32Array,
  mask: Float32Array,
  dropProb: number
): Float32Array {
  const output = new Float32Array(x.length);
  const scale = 1.0 / (1.0 - dropProb);

  for (let i = 0; i < x.length; i++) {
    output[i] = x[i] + residual[i] * mask[i] * scale;
  }

  return output;
}

export default residualAddRef;
