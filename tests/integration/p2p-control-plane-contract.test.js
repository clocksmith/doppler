import assert from 'node:assert/strict';

import {
  P2P_TRANSPORT_ERROR_CODES,
} from '../../src/experimental/distribution/p2p-transport-contract.js';
import {
  normalizeP2PPolicyDecision,
} from '../../src/experimental/distribution/p2p-control-plane.js';

for (const value of [undefined, null, {}]) {
  assert.throws(
    () => normalizeP2PPolicyDecision(value),
    (error) => error?.code === P2P_TRANSPORT_ERROR_CODES.payloadInvalid
  );
}

assert.throws(
  () => normalizeP2PPolicyDecision({ allow: true, deny: true }),
  (error) => error?.code === P2P_TRANSPORT_ERROR_CODES.payloadInvalid
);

assert.deepEqual(
  normalizeP2PPolicyDecision({ allow: false, reason: 'blocked' }),
  {
    allow: false,
    reason: 'blocked',
    sessionUpdate: {
      hasSessionToken: true,
      hasTokenExpiresAtMs: true,
      sessionToken: null,
      tokenExpiresAtMs: null,
      metadata: null,
    },
    metadata: null,
  }
);

console.log('p2p-control-plane-contract.test: ok');
