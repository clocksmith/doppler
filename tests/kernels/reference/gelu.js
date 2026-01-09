/**
 * GELU Reference Implementation
 *
 * GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */

const SQRT_2_PI = Math.sqrt(2 / Math.PI);
const GELU_COEFF = 0.044715;

/**
 * Standard GELU activation
 * @param {Float32Array} input
 * @returns {Float32Array}
 */
export function geluRef(input) {
  const output = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const x = input[i];
    const cdf = 0.5 * (1 + Math.tanh(SQRT_2_PI * (x + GELU_COEFF * x * x * x)));
    output[i] = x * cdf;
  }
  return output;
}

/**
 * Fast GELU approximation (used in some models)
 * GELU(x) ≈ x * sigmoid(1.702 * x)
 * @param {Float32Array} input
 * @returns {Float32Array}
 */
export function geluFastRef(input) {
  const output = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const x = input[i];
    output[i] = x / (1 + Math.exp(-1.702 * x));
  }
  return output;
}

/**
 * GeGLU activation: GELU(gate) * up
 * Used in some transformer FFN variants
 * @param {Float32Array} gate
 * @param {Float32Array} up
 * @returns {Float32Array}
 */
export function gegluRef(gate, up) {
  const output = new Float32Array(gate.length);
  for (let i = 0; i < gate.length; i++) {
    const x = gate[i];
    const cdf = 0.5 * (1 + Math.tanh(SQRT_2_PI * (x + GELU_COEFF * x * x * x)));
    const geluGate = x * cdf;
    output[i] = geluGate * up[i];
  }
  return output;
}
