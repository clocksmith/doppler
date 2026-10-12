import { GENERATION_CONTRACT } from './generation-contract.js';
import { contractAssert as assert, exactRecord, canonicalMetadata, admitTokens, hash,
  count, scalarString, immutableRecord, metadataDigest } from './canonical-contract-codec.js';

/** @type {import('./canonical-generation-policy.js').validateCanonicalGenerationPolicy} */
export function validateCanonicalGenerationPolicy(policy, vocabSize) {
  const fields = GENERATION_CONTRACT.canonical.policy;
  exactRecord(policy, Object.keys(fields));
  canonicalMetadata(policy);
  const record = /** @type {Record<string, unknown>} */ (policy);
  count(vocabSize, 'vocabulary', 0x100000000);
  assert(vocabSize > 0, 'Empty vocabulary.');
  for (const [key, rule] of Object.entries(fields)) {
    const value = record[key];
    if (rule.nullable && value === null) continue;
    if (rule.type === 'tokens') {
      admitTokens(value, vocabSize, GENERATION_CONTRACT.canonical.metadataLimits.maxNodes);
      assert(Array.isArray(value), `Invalid ${key}.`);
      assert(new Set(value).size === value.length, `${key} contains duplicate tokens.`);
    } else if (rule.type === 'token') count(value, key, vocabSize - 1);
    else if (rule.type === 'hash') hash(value, key);
    else if (rule.type === 'array') {
      assert(rule.maxItems !== undefined && rule.maxItemLength !== undefined, `Missing ${key} bounds.`);
      assert(Array.isArray(value) && value.length <= rule.maxItems, `Invalid ${key} array.`);
      for (const item of value) {
        scalarString(item);
        assert(item.length > 0 && item.length <= rule.maxItemLength, `Invalid ${key} item.`);
      }
    } else {
      assert(rule.type === 'integer' ? Number.isSafeInteger(value)
        : rule.type === 'number' ? Number.isFinite(value) : typeof value === rule.type, `Invalid ${key}.`);
      if (typeof value === 'number') assert((rule.minimum === undefined || value >= rule.minimum)
        && (rule.maximum === undefined || value <= rule.maximum)
        && (rule.exclusiveMinimum === undefined || value > rule.exclusiveMinimum), `Invalid ${key} range.`);
      assert(!rule.enum || rule.enum.includes(String(value)), `Unsupported ${key}.`);
      assert(!rule.pattern || new RegExp(rule.pattern).test(String(value)), `Invalid ${key} encoding.`);
    }
  }
  return immutableRecord(/** @type {import('./generation-contract.js').CanonicalGenerationPolicy} */ (policy));
}

// The admitted seed must be durably saved by the caller before this resolves.
// There is deliberately no time-based or numeric-seed fallback.
/** @type {import('./canonical-generation-policy.js').admitCanonicalGenerationPolicy} */
export async function admitCanonicalGenerationPolicy(input, vocabSize, persist) {
  assert(typeof persist === 'function', 'Generation policy persistence is required.');
  const fields = Object.keys(GENERATION_CONTRACT.canonical.policy);
  exactRecord(input, Object.hasOwn(input, 'seedHex64') ? fields : fields.filter(key => key !== 'seedHex64'));
  canonicalMetadata(input);
  let policy = { ...input };
  if (!Object.hasOwn(policy, 'seedHex64')) {
    const seed = globalThis.crypto.getRandomValues(new Uint32Array(2));
    policy.seedHex64 = [...seed].map(word => word.toString(16).padStart(8, '0')).join('');
  }
  const admitted = validateCanonicalGenerationPolicy(policy, vocabSize);
  await persist(admitted);
  return admitted;
}

/** @type {import('./canonical-generation-policy.js').canonicalGenerationIdentity} */
export function canonicalGenerationIdentity(policy, executionIdentity, vocabSize) {
  hash(executionIdentity, 'execution identity');
  const admitted = validateCanonicalGenerationPolicy(policy, vocabSize);
  const body = { policy: admitted, executionIdentity };
  return immutableRecord({ ...body, generationId: metadataDigest('doppler.generation-identity/v3', body) });
}
