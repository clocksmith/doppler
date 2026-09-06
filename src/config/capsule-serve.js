import schema from './capsule-serve.schema.json' with { type: 'json' };
import { assertCapsuleOperationFields, normalizeCapsuleObservation } from './capsule-operation.js';
import { freezeCapsuleV2 } from './capsule-v2.js';

export function normalizeCapsuleServePolicy(value) {
  const policy = normalizeCapsuleObservation(value);
  assertCapsuleOperationFields(policy, schema.required, 'Capsule serving policy');
  if (schema.required.some(key => !Object.hasOwn(policy, key)) || policy.schema !== schema.properties.schema.const) {
    throw new Error('Capsule serving requires a complete doppler.capsule-serve/v1 policy.');
  }
  for (const [key, rule] of Object.entries(schema.properties)) {
    if (rule.type !== 'integer') continue;
    if (!Number.isSafeInteger(policy[key]) || policy[key] < rule.minimum
      || (rule.maximum !== undefined && policy[key] > rule.maximum)) {
      throw new Error(`Invalid Capsule serving policy ${key}.`);
    }
  }
  if (!Array.isArray(policy.allowedOrigins) || new Set(policy.allowedOrigins).size !== policy.allowedOrigins.length) {
    throw new Error('Capsule serving allowedOrigins must be a unique array of explicit origins.');
  }
  for (const origin of policy.allowedOrigins) {
    const url = new URL(origin);
    if (typeof origin !== 'string' || !['http:', 'https:'].includes(url.protocol) || url.origin !== origin) {
      throw new Error('Capsule serving requires exact HTTP(S) origins; wildcards and opaque origins are forbidden.');
    }
  }
  return freezeCapsuleV2(policy);
}
