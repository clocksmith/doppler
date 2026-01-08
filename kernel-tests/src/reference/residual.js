/**
 * Reference Residual Add Implementation
 */

/**
 * Reference residual add
 * output = x + residual
 */
export function residualAddRef(x, residual) {
  const output = new Float32Array(x.length);

  for (let i = 0; i < x.length; i++) {
    output[i] = x[i] + residual[i];
  }

  return output;
}

/**
 * In-place residual add (modifies x)
 */
export function residualAddInplaceRef(x, residual) {
  for (let i = 0; i < x.length; i++) {
    x[i] += residual[i];
  }
  return x;
}

/**
 * Scaled residual add
 * output = x + scale * residual
 */
export function scaledResidualAddRef(x, residual, scale) {
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
export function residualAddDropoutRef(x, residual, mask, dropProb) {
  const output = new Float32Array(x.length);
  const scale = 1.0 / (1.0 - dropProb);

  for (let i = 0; i < x.length; i++) {
    output[i] = x[i] + residual[i] * mask[i] * scale;
  }

  return output;
}

export default residualAddRef;
