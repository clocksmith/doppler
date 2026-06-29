import assert from 'node:assert/strict';

import {
  P2P_TRANSPORT_ERROR_CODES,
} from '../../src/experimental/distribution/p2p-transport-contract.js';
import {
  assertPeerSupportsDescriptor,
  createDescriptorPeerAssignment,
  negotiateDescriptorShardCache,
  normalizePeerCapabilityProfile,
  validateActivationTransportPayload,
} from '../../src/experimental/distribution/descriptor-transport.js';

const digestA = `sha256:${'a'.repeat(64)}`;
const digestB = `sha256:${'b'.repeat(64)}`;
const digestC = `sha256:${'c'.repeat(64)}`;
const digestD = `sha256:${'d'.repeat(64)}`;

const descriptorManifest = {
  descriptor_hash: digestA,
  components: {
    prng_substrate: {
      algorithm: 'coord_hash_normal_v1',
    },
    kronecker_sum: {
      shard_file: 'layer0.kron',
    },
    coordinate_inr: {
      type: 'siren',
      shard_file: 'layer0.siren',
    },
    sparse_outliers: {
      format: 'coo_v1',
      shard_file: 'layer0.sparse',
    },
  },
};

const profile = normalizePeerCapabilityProfile({
  available_vram_bytes: 8589934592,
  backends: ['webgpu', 'metal'],
  supported_generators: ['coord_hash_normal_v1', 'siren_f16_v1'],
  bandwidth_bps: 12500000,
  latency_ms: 45,
  reliability_score: 0.98,
});

assert.deepEqual(profile.backends, ['webgpu', 'metal']);

assert.deepEqual(
  assertPeerSupportsDescriptor(profile, descriptorManifest).requiredGenerators,
  ['coord_hash_normal_v1', 'siren_f16_v1']
);

assert.throws(
  () => assertPeerSupportsDescriptor(
    { ...profile, supported_generators: ['coord_hash_normal_v1'] },
    descriptorManifest
  ),
  (error) => error?.code === P2P_TRANSPORT_ERROR_CODES.policyDenied
);

const ready = negotiateDescriptorShardCache({
  descriptorManifest,
  descriptorShardHashes: {
    'layer0.kron': digestB,
    'layer0.siren': digestC,
    'layer0.sparse': digestD,
  },
  peerDescriptorCache: {
    descriptorHash: digestA,
    shards: {
      'layer0.kron': digestB,
      'layer0.siren': { hash: digestC },
      'layer0.sparse': digestD,
    },
  },
});
assert.equal(ready.ready, true);
assert.equal(ready.missingShards.length, 0);

const mismatch = negotiateDescriptorShardCache({
  descriptorManifest,
  peerDescriptorCache: {
    descriptorHash: digestB,
    shards: {
      'layer0.kron': digestB,
    },
  },
});
assert.equal(mismatch.ready, false);
assert.equal(mismatch.missingShards.length, 3);
assert.equal(mismatch.missingShards[0].reason, 'descriptor_hash_mismatch');

const assignment = createDescriptorPeerAssignment({
  descriptorManifest,
  descriptorShardHashes: {
    'layer0.kron': digestB,
    'layer0.siren': digestC,
    'layer0.sparse': digestD,
  },
  peerCapabilityProfile: profile,
  peerDescriptorCache: {
    descriptorHash: digestA,
    shards: {
      'layer0.kron': digestB,
      'layer0.siren': digestC,
      'layer0.sparse': digestD,
    },
  },
  activationPayload: new Uint8Array(8192),
  modelDim: 4096,
  tokenCount: 1,
  requiredVramBytes: 1024,
});
assert.equal(assignment.assignable, true);
assert.equal(assignment.requiredDownloads.length, 0);
assert.equal(assignment.activation.expectedBytes, 8192);

const needsDownload = createDescriptorPeerAssignment({
  descriptorManifest,
  peerCapabilityProfile: profile,
  peerDescriptorCache: {
    descriptorHash: digestB,
    shards: {},
  },
  requiredVramBytes: 1024,
});
assert.equal(needsDownload.assignable, false);
assert.equal(needsDownload.blockers[0].code, 'descriptor_shards_missing');
assert.equal(needsDownload.requiredDownloads.length, 3);

assert.throws(
  () => createDescriptorPeerAssignment({
    descriptorManifest,
    peerCapabilityProfile: profile,
    peerDescriptorCache: {
      descriptorHash: digestB,
      shards: {},
    },
    requiredVramBytes: 1024,
    failClosed: true,
  }),
  (error) => error?.code === P2P_TRANSPORT_ERROR_CODES.policyDenied
);

assert.deepEqual(
  validateActivationTransportPayload(new Uint8Array(8192), {
    modelDim: 4096,
    tokenCount: 1,
  }),
  {
    contractVersion: 1,
    modelDim: 4096,
    tokenCount: 1,
    bytesPerToken: 8192,
    expectedBytes: 8192,
    actualBytes: 8192,
  }
);

assert.throws(
  () => validateActivationTransportPayload(new Uint8Array(8191), {
    modelDim: 4096,
    tokenCount: 1,
  }),
  (error) => error?.code === P2P_TRANSPORT_ERROR_CODES.payloadInvalid
);

console.log('descriptor-transport-contract.test: ok');
