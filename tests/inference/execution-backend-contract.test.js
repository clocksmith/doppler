import assert from 'node:assert/strict';
import { validateExecutionBackend, assertExecutionBackendAvailable, executionStateNamespace } from '../../src/inference/execution-backend-contract.js';
import { buildProviderPrivacyReceipt } from '../../src/client/model-host/privacy-receipt.js';

const digest = c => 'sha256:' + c.repeat(64);
const backend = { schema: 'doppler.execution-backend/v1', id: 'doppler-plaintext/v1', version: 1, kind: 'plaintext',
  modelIdentities: [digest('a')], operations: ['generate'], protocol: null, inputRepresentation: 'messages', outputRepresentation: 'text' };
assert(Object.isFrozen(validateExecutionBackend(backend)));
assert.equal(assertExecutionBackendAvailable(backend, digest('a')).kind, 'plaintext');
assert.throws(() => assertExecutionBackendAvailable(backend, digest('b')), /exact model/);
assert.throws(() => validateExecutionBackend({ ...backend, inputRepresentation: 'opaque' }), /Plaintext/);
for (const kind of ['attested', 'cryptographic']) {
  const claim = { ...backend, kind, protocol: kind + '/v1', inputRepresentation: 'opaque', outputRepresentation: 'opaque' };
  assert.equal(validateExecutionBackend(claim).kind, kind);
  assert.throws(() => assertExecutionBackendAvailable(claim, digest('a')), /unavailable/);
}
const state = { schema: 'doppler.execution-state/v1', modelIdentity: digest('a'), adapterIdentities: [], backendId: backend.id, policyId: digest('b'),
  routeId: digest('c'), tenantId: 'tenant-a', sessionId: 'thread-a', keyContext: null };
const namespace = executionStateNamespace(state);
for (const change of [{ tenantId: 'tenant-b' }, { sessionId: 'thread-b' }, { keyContext: 'other-key' },
  { adapterIdentities: [digest('d')] }, { modelIdentity: digest('d') }, { policyId: digest('d') }, { routeId: digest('d') }, { backendId: 'other/v1' }]) {
  assert.notEqual(executionStateNamespace({ ...state, ...change }), namespace);
}
assert.throws(() => executionStateNamespace({ ...state, tenantId: '' }));
const receipt = buildProviderPrivacyReceipt({ attemptId: 'attempt', policyId: state.policyId, routeId: state.routeId,
  modelIdentity: state.modelIdentity, backendId: state.backendId, status: 'completed',
  outputText: 'private output', tokenIds: [123], transcript: 'private transcript', inputHash: digest('e'),
  stats: { private: 'private prompt' }, error: 'private error' });
for (const key of ['outputText', 'tokenIds', 'transcript', 'inputHash', 'stats', 'error']) assert(!Object.hasOwn(receipt, key));
assert(!JSON.stringify(receipt).includes('private'));
assert.equal(receipt.confidentialityProof, false);
assert.equal(receipt.integrityProof, false);
assert.equal(receipt.availabilityProof, false);
assert(Object.isFrozen(receipt));
console.log('execution-backend-contract: exact support, untrusted claims, state isolation and content-free receipts passed');
