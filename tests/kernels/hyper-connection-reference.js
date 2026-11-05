// Float64 reference of the pinned qwen4_exp gated residual
// (artifacts/qwen38-flash-next-20261013/source/modeling_qwen4_exp.py,
// Qwen4ExpTextGatedResidual and Qwen4ExpTextDecoderLayer injection) with
// first-order forward error bounds for F32 execution in any summation order.
// Test-only: never a runtime fallback.

export const F32_UNIT_ROUNDOFF = 2 ** -24;
const u = F32_UNIT_ROUNDOFF;
const gamma = (n) => (n * u) / (1 - n * u);
// max |silu'(x)| over the reals; max |sigmoid'(x)| = 1/4.
export const SILU_LIPSCHITZ = 1.0998;
// Relative error budget of the refined sigmoid/SiLU helpers (exp, reciprocal, product).
const ACTIVATION_RELATIVE = 8 * u;
// WGSL may flush subnormal f32 results to zero; activation outputs allow an
// absolute error of the smallest normal f32 (relevant only below 2^-126).
export const F32_MIN_NORMAL = 2 ** -126;

const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const silu = (x) => x * sigmoid(x);

/**
 * Deterministic recipes of artifacts/qwen38-flash-next-20261013/prepare-reference.py:
 * parameter p (ordinal o) = ((i * (o + 1)) % 31 - 15) / 4096; input = ((i % 43) - 21) / 16.
 */
export function fixtureParameter(count, ordinal) {
  return Float32Array.from({ length: count }, (_, i) => (((i * (ordinal + 1)) % 31) - 15) / 4096);
}
export function fixtureInput(count) {
  return Float32Array.from({ length: count }, (_, i) => ((i % 43) - 21) / 16);
}

/** y[t, n] = sum_k x[t, k] * w[n, k] in float64 with |x||w| for error bounds. */
function linear(x, xBound, weight, tokens, inFeatures, outFeatures) {
  const value = new Float64Array(tokens * outFeatures);
  const bound = new Float64Array(tokens * outFeatures);
  const g = gamma(inFeatures);
  for (let t = 0; t < tokens; t++) {
    for (let n = 0; n < outFeatures; n++) {
      let sum = 0;
      let absolute = 0;
      let propagated = 0;
      const row = n * inFeatures;
      for (let k = 0; k < inFeatures; k++) {
        const xv = x[t * inFeatures + k];
        const wv = weight[row + k];
        sum += xv * wv;
        absolute += Math.abs(xv * wv);
        propagated += Math.abs(wv) * xBound[t * inFeatures + k];
      }
      value[t * outFeatures + n] = sum;
      bound[t * outFeatures + n] = g * (absolute + propagated) + propagated;
    }
  }
  return { value, bound };
}

/** Grouped RMSNorm with (1 + weight): value and elementwise bound. */
export function groupedNormReference(x, weight, { tokens, streams, width, eps }) {
  const value = new Float64Array(tokens * streams * width);
  const bound = new Float64Array(value.length);
  // Sum of squares (positive terms, squares possibly rounded): gamma_{W+1}; mean
  // division, eps addition: 2u; inverse root halves the relative argument error
  // and adds 2u; two products and (1 + w): 3u.
  const relative = 0.5 * (gamma(width + 1) + 2 * u) + 2 * u + 3 * u;
  for (let group = 0; group < tokens * streams; group++) {
    const base = group * width;
    const weightBase = (group % streams) * width;
    let sum = 0;
    for (let i = 0; i < width; i++) sum += x[base + i] * x[base + i];
    const inverseRms = 1 / Math.sqrt(sum / width + eps);
    for (let i = 0; i < width; i++) {
      const normalized = x[base + i] * inverseRms * (1 + weight[weightBase + i]);
      value[base + i] = normalized;
      bound[base + i] = Math.abs(normalized) * relative;
    }
  }
  return { value, bound };
}

/** Elementwise activation values and bounds given input values and input bounds. */
export function activationReference(mode, input, inputBound, streams) {
  const value = new Float64Array(input.length);
  const bound = new Float64Array(input.length);
  for (let i = 0; i < input.length; i++) {
    if (mode === 'low_rank_silu') {
      const scaled = input[i] / streams;
      value[i] = silu(scaled);
      bound[i] = SILU_LIPSCHITZ * (inputBound[i] / streams + u * Math.abs(scaled)) + ACTIVATION_RELATIVE * Math.abs(value[i]) + F32_MIN_NORMAL;
    } else if (mode === 'mix_sigmoid') {
      value[i] = sigmoid(input[i]);
      bound[i] = 0.25 * inputBound[i] + ACTIVATION_RELATIVE * value[i] + F32_MIN_NORMAL;
    } else if (mode === 'inject_gate') {
      const scaled = input[i] / streams;
      value[i] = 2 * sigmoid(scaled);
      bound[i] = 2 * (0.25 * (inputBound[i] / streams + u * Math.abs(scaled))) + ACTIVATION_RELATIVE * value[i] + F32_MIN_NORMAL;
    } else {
      throw new Error(`unknown activation ${mode}`);
    }
  }
  return { value, bound };
}

/** Branch input: mean over streams of mix * normed. */
export function branchReference(mix, mixBound, normed, normedBound, { tokens, streams, width }) {
  const value = new Float64Array(tokens * width);
  const bound = new Float64Array(tokens * width);
  for (let t = 0; t < tokens; t++) {
    for (let w = 0; w < width; w++) {
      let sum = 0;
      let absolute = 0;
      let propagated = 0;
      for (let s = 0; s < streams; s++) {
        const index = (t * streams + s) * width + w;
        sum += mix[index] * normed[index];
        absolute += Math.abs(mix[index] * normed[index]);
        propagated += Math.abs(normed[index]) * mixBound[index] + Math.abs(mix[index]) * normedBound[index];
      }
      value[t * width + w] = sum / streams;
      bound[t * width + w] = (gamma(streams + 1) * absolute + propagated) / streams + u * Math.abs(sum / streams);
    }
  }
  return { value, bound };
}

/** Injection residual: streams + branch * gate[stream]. */
export function injectReference(streams, branch, branchBound, gate, gateBound, { tokens, streamCount, width }) {
  const value = new Float64Array(tokens * streamCount * width);
  const bound = new Float64Array(value.length);
  for (let t = 0; t < tokens; t++) {
    for (let s = 0; s < streamCount; s++) {
      for (let w = 0; w < width; w++) {
        const index = (t * streamCount + s) * width + w;
        const b = branch[t * width + w];
        const g = gate[t * streamCount + s];
        const product = b * g;
        value[index] = streams[index] + product;
        bound[index] = Math.abs(g) * branchBound[t * width + w] + Math.abs(b) * gateBound[t * streamCount + s]
          + u * Math.abs(product) * 2 + u * Math.abs(value[index]);
      }
    }
  }
  return { value, bound };
}

/**
 * Full Qwen4ExpTextGatedResidual: returns every intermediate with bounds.
 * weights: norm [S*W], down [R, S*W], up [S*W, R], inject [S, S*W] or null (root mixer).
 */
export function gatedResidualReference(x, weights, { tokens, streams, width, rank, eps }) {
  const hc = streams * width;
  const exact = new Float64Array(x.length);
  const normed = groupedNormReference(x, weights.norm, { tokens, streams, width, eps });
  const down = linear(normed.value, normed.bound, weights.down, tokens, hc, rank);
  const lowRank = activationReference('low_rank_silu', down.value, down.bound, streams);
  const up = linear(lowRank.value, lowRank.bound, weights.up, tokens, rank, hc);
  const mix = activationReference('mix_sigmoid', up.value, up.bound, streams);
  const branch = branchReference(mix.value, mix.bound, normed.value, normed.bound, { tokens, streams, width });
  let injectLogits = null;
  let injection = null;
  if (weights.inject) {
    injectLogits = linear(normed.value, normed.bound, weights.inject, tokens, hc, streams);
    injection = activationReference('inject_gate', injectLogits.value, injectLogits.bound, streams);
  }
  exact.set(x);
  return { normed, down, lowRank, up, mix, branch, injectLogits, injection, streams: exact };
}

/** Max |actual - expected| / bound and the worst index; ratio <= 1 passes. */
export function boundRatio(actual, expected, bound) {
  let worst = 0;
  let worstIndex = -1;
  for (let i = 0; i < expected.length; i++) {
    const error = Math.abs(actual[i] - expected[i]);
    const ratio = bound[i] > 0 ? error / bound[i] : (error === 0 ? 0 : Infinity);
    if (!(ratio <= worst)) {
      worst = ratio;
      worstIndex = i;
    }
  }
  return { ratio: worst, index: worstIndex };
}
