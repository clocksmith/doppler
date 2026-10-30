/** @type {typeof import('./fp8.schema.js').validateFP8Policy} */
export function validateFP8Policy(input, label = 'FP8 policy') {
  if (!input || typeof input !== 'object' || Array.isArray(input)) {
    throw new Error(`${label} requires an explicit doppler.fp8/v1 policy.`);
  }
  const policy = /** @type {Record<string, unknown>} */ (input);
  const fixed = { schema: 'doppler.fp8/v1', packing: 'fp8-block-v1', scaleSemantics: 'multiply',
    scaleDtype: 'f32', inputDtype: 'f32', accumulatorDtype: 'f32', outputDtype: 'f32', dequantizedWeightPolicy: 'finite-normal-or-zero', nonfinite: 'preserve-storage-reject-gpu' };
  const keys = [...Object.keys(fixed), 'encoding', 'blockShape', 'scaleMode', 'scaleSuffix'];
  if (Object.keys(policy).length !== keys.length || keys.some(key => !(key in policy))) {
    throw new Error(`${label} must explicitly declare exactly ${keys.join(', ')}.`);
  }
  for (const [key, value] of Object.entries(fixed)) {
    if (policy[key] !== value) throw new Error(`${label}.${key} must be ${value}.`);
  }
  if (!['e4m3fn', 'e5m2'].includes(String(policy.encoding))) throw new Error(`${label}.encoding is unsupported.`);
  const shape = policy.blockShape;
  if (!Array.isArray(shape) || shape.length !== 2
    || !shape.every(value => Number.isSafeInteger(value) && value > 0 && value <= 65536)
    || shape[0] * shape[1] > 65536) throw new Error(`${label}.blockShape requires two positive dimensions with at most65536 elements.`);
  if ((policy.scaleMode === 'identity' && policy.scaleSuffix !== null)
    || (policy.scaleMode === 'companion' && (typeof policy.scaleSuffix !== 'string' || ! /^[._]/.test(policy.scaleSuffix) || policy.scaleSuffix.length < 2))
    || !['identity', 'companion'].includes(String(policy.scaleMode))) {
    throw new Error(`${label} requires identity/null or companion/explicit scaleSuffix.`);
  }
  return /** @type {import('./fp8.schema.js').FP8Policy} */ (input);
}
