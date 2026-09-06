import schema from './pack-serve.schema.json' with { type: 'json' };
import { assertPackOperationFields, normalizePackObservation } from './pack-operation.js';
import { freezePackV2 } from './pack-v2.js';

export function normalizePackServePolicy(value) {
  const policy = normalizePackObservation(value);
  assertPackOperationFields(policy, schema.required, 'Pack serving policy');
  if (schema.required.some(key => !Object.hasOwn(policy, key)) || policy.schema !== schema.properties.schema.const) {
    throw new Error('Pack serving requires a complete doppler.pack-serve/v1 policy.');
  }
  for (const [key, rule] of Object.entries(schema.properties)) {
    if (rule.type !== 'integer') continue;
    if (!Number.isSafeInteger(policy[key]) || policy[key] < rule.minimum
      || (rule.maximum !== undefined && policy[key] > rule.maximum)) {
      throw new Error(`Invalid Pack serving policy ${key}.`);
    }
  }
  if (!Array.isArray(policy.allowedOrigins) || new Set(policy.allowedOrigins).size !== policy.allowedOrigins.length) {
    throw new Error('Pack serving allowedOrigins must be a unique array of explicit origins.');
  }
  for (const origin of policy.allowedOrigins) {
    const url = new URL(origin);
    if (typeof origin !== 'string' || !['http:', 'https:'].includes(url.protocol) || url.origin !== origin) {
      throw new Error('Pack serving requires exact HTTP(S) origins; wildcards and opaque origins are forbidden.');
    }
  }
  return freezePackV2(policy);
}
