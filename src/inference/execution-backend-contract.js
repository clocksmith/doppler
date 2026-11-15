/** Data/operation boundary only. Privacy qualification is never inferred from a remote descriptor. */
import { contractAssert as assert, exactRecord, canonicalMetadata, hash, immutableRecord, metadataDigest } from '../config/canonical-contract-codec.js';

/** @type {import('./execution-backend-contract.js').validateExecutionBackend} */
export function validateExecutionBackend(input) {
  canonicalMetadata(input);
  const value = exactRecord(input, ['schema', 'id', 'version', 'kind', 'modelIdentities', 'operations', 'protocol', 'inputRepresentation', 'outputRepresentation']);
  assert(value.schema === 'doppler.execution-backend/v1' && typeof value.id === 'string' && value.id.length > 0
    && value.id.length <= 256 && value.version === 1, 'Versioned execution backend required.');
  assert(['plaintext', 'attested', 'cryptographic'].includes(value.kind), 'Unknown backend kind.');
  assert(Array.isArray(value.modelIdentities) && value.modelIdentities.length > 0 && value.modelIdentities.length <= 128,
    'Explicit backend model identities required.');
  value.modelIdentities.forEach(identity => hash(identity, 'backend model'));
  assert(new Set(value.modelIdentities).size === value.modelIdentities.length, 'Duplicate backend model identity.');
  assert(Array.isArray(value.operations) && value.operations.length === 1 && value.operations[0] === 'generate',
    'Explicit supported backend operation required.');
  if (value.kind === 'plaintext') {
    assert(value.protocol === null && value.inputRepresentation === 'messages' && value.outputRepresentation === 'text',
      'Plaintext backend cannot claim ciphertext or an encrypted protocol.');
  } else {
    assert(typeof value.protocol === 'string' && value.protocol.length > 0 && value.protocol.length <= 256,
      'Protected backend requires a protocol identity.');
    assert(value.inputRepresentation === 'opaque' && value.outputRepresentation === 'opaque',
      'Protected backend requires a separate opaque transport contract.');
  }
  return immutableRecord(value);
}

/** No attested verifier or encrypted evaluator is installed. A descriptor is not executable qualification.
 * @type {import('./execution-backend-contract.js').assertExecutionBackendAvailable} */
export function assertExecutionBackendAvailable(input, modelIdentity) {
  const value = validateExecutionBackend(input);
  hash(modelIdentity, 'requested model');
  assert(value.modelIdentities.includes(modelIdentity), 'Backend does not support this exact model.');
  assert(value.kind === 'plaintext', 'Attested and cryptographic backends are unavailable; descriptor claims cannot enable them.');
  return value;
}

/** State identity contains authorization/key identifiers, never prompt/output digests.
 * @type {import('./execution-backend-contract.js').executionStateNamespace} */
export function executionStateNamespace(input) {
  const value = exactRecord(input, ['schema', 'modelIdentity', 'adapterIdentities', 'backendId', 'policyId', 'routeId', 'tenantId', 'sessionId', 'keyContext']);
  assert(value.schema === 'doppler.execution-state/v1', 'Versioned privacy state identity required.');
  for (const identity of [value.modelIdentity, value.policyId, value.routeId]) hash(identity, 'state binding');
  assert(Array.isArray(value.adapterIdentities) && value.adapterIdentities.length <= 128, 'Explicit state adapter identities required.');
  value.adapterIdentities.forEach(identity => hash(identity, 'state adapter'));
  for (const id of [value.backendId, value.tenantId, value.sessionId]) {
    assert(typeof id === 'string' && id.length > 0 && id.length <= 256, 'Bounded state identity required.');
  }
  assert(value.keyContext === null || typeof value.keyContext === 'string' && value.keyContext.length > 0
    && value.keyContext.length <= 256, 'Explicit state key context required.');
  return metadataDigest(value.schema, value);
}
