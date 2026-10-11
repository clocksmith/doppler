// Versioned arithmetic and storage contract, never an inferred runtime default.
/** @type {Readonly<import('./w8a8.schema.js').W8A8Policy>} */
export const W8A8_POLICY = Object.freeze({
  schema: 'doppler.w8a8/v1',
  scheme: 'symmetric',
  weightGranularity: 'output-channel',
  activationGranularity: 'token',
  codeRange: Object.freeze(/** @type {const} */ ([-127, 127])),
  inputDtype: 'f32',
  scaleDtype: 'f32',
  zeroScale: 1,
  rounding: 'nearest-even',
  inputPolicy: 'finite-normal-or-zero',
  scalePolicy: 'positive-normal',
  packing: 'w8a8-row-v1',
  accumulatorDtype: 'i32',
  maxReductionWidth: 133144,
  rescaleOrder: Object.freeze(/** @type {const} */ (['activation', 'weight'])),
  outputDtype: 'f32',
  quantizationRatio: 'exact-input-over-stored-scale',
  scaleRounding: 'nearest-even',
  rescaleRounding: 'nearest-even',
  outputSubnormals: 'flush-to-signed-zero-after-each-rescale',
});

/** @param {unknown} policy @param {string} [label] @returns {import('./w8a8.schema.js').W8A8Policy} */
export function validateW8A8Policy(policy, label = 'W8A8 policy') {
  if (!policy || typeof policy !== 'object' || Array.isArray(policy)
    || Object.keys(policy).length !== Object.keys(W8A8_POLICY).length) {
    throw new Error(`${label} must explicitly declare the complete doppler.w8a8/v1 policy.`);
  }
  for (const [key, expected] of Object.entries(W8A8_POLICY)) {
    if (JSON.stringify(/** @type {Record<string, unknown>} */ (policy)[key]) !== JSON.stringify(expected)) {
      throw new Error(`${label}.${key} must be ${JSON.stringify(expected)}.`);
    }
  }
  return /** @type {import('./w8a8.schema.js').W8A8Policy} */ (policy);
}
