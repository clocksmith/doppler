import assert from 'node:assert/strict';

const {
  P2P_TRANSPORT_ERROR_CODES,
  normalizeP2PTransportError,
  normalizeP2PTransportResult,
  isP2PTransportRetryable,
} = await import('../../src/distribution/p2p-transport-contract.js');

{
  const result = normalizeP2PTransportResult({
    data: new Uint8Array([1, 2, 3]),
    manifestVersionSet: 'manifest:v1:test',
    manifestHash: 'sha256:abcd',
  });
  assert.equal(result?.schemaVersion, 1);
  assert.equal(result?.data?.byteLength, 3);
  assert.equal(result?.manifestVersionSet, 'manifest:v1:test');
  assert.equal(result?.manifestHash, 'sha256:abcd');
}

{
  const unavailable = normalizeP2PTransportError(new Error('peer miss'));
  assert.equal(unavailable.code, P2P_TRANSPORT_ERROR_CODES.unavailable);
  assert.equal(isP2PTransportRetryable(unavailable), false);
}

{
  const denied = normalizeP2PTransportError(new Error('policy denied by remote peer'));
  assert.equal(denied.code, P2P_TRANSPORT_ERROR_CODES.policyDenied);
  assert.equal(isP2PTransportRetryable(denied), false);
}

{
  const timeout = normalizeP2PTransportError(new Error('timed out waiting for chunk'));
  assert.equal(timeout.code, P2P_TRANSPORT_ERROR_CODES.timeout);
  assert.equal(isP2PTransportRetryable(timeout), true);
}

{
  const internal = normalizeP2PTransportError(new Error('unexpected transport failure'));
  assert.equal(internal.code, P2P_TRANSPORT_ERROR_CODES.internal);
  assert.equal(isP2PTransportRetryable(internal), true);
}

console.log('p2p-transport-contract.test: ok');
