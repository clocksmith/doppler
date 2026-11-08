import assert from 'node:assert/strict';
import { resolveShardHashVerification, resolveKVCacheAllocation } from '../../src/inference/pipelines/text/load-policy.js';

// The runtime config owns hash verification. A storage context created without
// an explicit decision used to report verifyHashes=false and that value shadowed
// runtime.loading.shardCache.verifyHashes, so URL and file-path loads skipped
// every shard digest check by default.
const loadingConfig = { shardCache: { verifyHashes: true } };

assert.deepEqual(
  resolveShardHashVerification({ loadingConfig }),
  { verifyHashes: true, source: 'runtime.loading.shardCache.verifyHashes' }
);
assert.deepEqual(
  resolveShardHashVerification({ options: { verifyHashes: false }, loadingConfig }),
  { verifyHashes: false, source: 'options.verifyHashes' }
);
assert.throws(() => resolveShardHashVerification({ loadingConfig: { shardCache: {} } }), /verifyHashes is required/);
assert.throws(() => resolveShardHashVerification({}), /verifyHashes is required/);

// A context that verified its bytes already (OPFS after a verified download)
// satisfies the policy; the loader does not hash the shards a second time.
assert.deepEqual(
  resolveShardHashVerification({ loadingConfig, storageContext: { verifyHashes: false, hashesTrusted: true } }),
  { verifyHashes: false, source: 'storageContext.hashesTrusted' }
);
// A context that merely disabled verification cannot relax the policy.
assert.throws(
  () => resolveShardHashVerification({ loadingConfig, storageContext: { verifyHashes: false } }),
  /created with verifyHashes=false/
);
// A verifying context under a verifying policy is the ordinary case.
assert.deepEqual(
  resolveShardHashVerification({ loadingConfig, storageContext: { verifyHashes: true } }),
  { verifyHashes: true, source: 'runtime.loading.shardCache.verifyHashes' }
);
// When the policy is off, the context's own setting is irrelevant to the loader.
assert.deepEqual(
  resolveShardHashVerification({ loadingConfig: { shardCache: { verifyHashes: false } }, storageContext: { verifyHashes: true } }),
  { verifyHashes: false, source: 'runtime.loading.shardCache.verifyHashes' }
);

// Embedding models that skip KV writes allocate no KV cache; everything else does.
const skip = { inference: { session: { skipEmbeddingKVCacheWrites: true } } };
const keep = { inference: { session: { skipEmbeddingKVCacheWrites: false } } };
assert.equal(resolveKVCacheAllocation({ modelType: 'embedding' }, skip).allocate, false);
assert.equal(resolveKVCacheAllocation({ modelType: 'Embedding' }, skip).allocate, false);
assert.match(resolveKVCacheAllocation({ modelType: 'embedding' }, skip).reason, /skipEmbeddingKVCacheWrites=true/);
assert.deepEqual(resolveKVCacheAllocation({ modelType: 'embedding' }, keep), { allocate: true, reason: null });
assert.deepEqual(resolveKVCacheAllocation({ modelType: 'transformer' }, skip), { allocate: true, reason: null });
assert.deepEqual(resolveKVCacheAllocation({ modelType: 'transformer', inference: { supportsRerank: true } }, skip), { allocate: true, reason: null });
assert.deepEqual(resolveKVCacheAllocation(null, skip), { allocate: true, reason: null });
console.log('shard hash policy and kv allocation ok');
