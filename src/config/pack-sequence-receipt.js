import { computeCanonicalSha256 } from '../formats/canonical-hash.js';

// Observation encoding only. This does not perform model tensor arithmetic.
function observation(value) {
  if (ArrayBuffer.isView(value)) return Array.from(value);
  if (Array.isArray(value)) return value.map(observation);
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, observation(item)]));
  return value;
}

export function hashPackSequenceInput(sequence, options) {
  const { signal, ...request } = options;
  return computeCanonicalSha256(observation({ sequence, options: request }));
}

export function hashPackSequenceOutput(result) {
  const { phase, receipt, dopplerProviderReceipt, ...output } = result;
  return computeCanonicalSha256(observation(output));
}
