import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

const config = JSON.parse(
  await fs.readFile(new URL('../../benchmarks/vendors/embedding-compare.config.json', import.meta.url), 'utf8')
);
const catalog = JSON.parse(
  await fs.readFile(new URL('../../models/catalog.json', import.meta.url), 'utf8')
);
const qwenEmbeddingMetalThroughputProfile = JSON.parse(
  await fs.readFile(
    new URL('../../src/config/runtime/profiles/qwen-3-embedding-0-6b-metal-throughput.json', import.meta.url),
    'utf8'
  )
);
const qwenEmbeddingMetalStabilityProfile = JSON.parse(
  await fs.readFile(
    new URL('../../src/config/runtime/profiles/qwen-3-embedding-0-6b-metal-stability.json', import.meta.url),
    'utf8'
  )
);

const catalogByModelId = new Map(
  (Array.isArray(catalog?.models) ? catalog.models : [])
    .filter((entry) => typeof entry?.modelId === 'string' && entry.modelId.length > 0)
    .map((entry) => [entry.modelId, entry])
);

assert.equal(config.schemaVersion, 1);
assert.ok(Array.isArray(config.modelProfiles) && config.modelProfiles.length > 0);

for (const profile of config.modelProfiles) {
  const modelId = profile?.dopplerModelId;
  assert.equal(typeof modelId, 'string', 'embedding compare profiles must define dopplerModelId');
  const catalogEntry = catalogByModelId.get(modelId);
  assert.ok(catalogEntry, `${modelId}: embedding compare profile must exist in models/catalog.json`);
  assert.equal(
    Array.isArray(catalogEntry.modes) && catalogEntry.modes.includes('embedding'),
    true,
    `${modelId}: embedding compare profiles must target embedding catalog entries`
  );
  assert.equal(profile.compareLane, 'performance_comparable', `${modelId}: embedding compare lane`);
  assert.equal(typeof profile.releaseClaimable, 'boolean', `${modelId}: releaseClaimable must be explicit`);
  assert.equal(
    typeof catalogEntry.vendorBenchmark?.transformersjs?.repoId,
    'string',
    `${modelId}: catalog must own the Transformers.js repo mapping`
  );
  assert.equal(
    typeof catalogEntry.vendorBenchmark?.transformersjs?.dtype,
    'string',
    `${modelId}: catalog must own the Transformers.js dtype mapping`
  );

  if (profile.releaseClaimable === true) {
    assert.equal(
      profile.defaultDopplerSource,
      'quickstart-registry',
      `${modelId}: release-claimable embedding compares must measure a hosted artifact`
    );
    assert.equal(catalogEntry.lifecycle?.availability?.hf, true, `${modelId}: release-claimable embedding compares require hosted HF availability`);
    assert.equal(typeof catalogEntry.hf?.revision, 'string', `${modelId}: release-claimable embedding compares require pinned HF revision`);
    assert.equal(catalogEntry.artifactCompleteness, 'complete', `${modelId}: release-claimable embedding compares require complete artifacts`);
    assert.equal(catalogEntry.runtimePromotionState, 'manifest-owned', `${modelId}: release-claimable embedding compares require manifest-owned runtime metadata`);
    assert.equal(catalogEntry.weightsRefAllowed, false, `${modelId}: release-claimable embedding compares must not depend on weightsRef fallback`);
  } else {
    assert.notEqual(
      profile.defaultDopplerSource,
      'quickstart-registry',
      `${modelId}: non-release embedding compares should not default to the hosted claim lane`
    );
  }
}

const qwenEmbedding = config.modelProfiles.find(
  (entry) => entry.dopplerModelId === 'qwen-3-embedding-0-6b-q4k-ehf16-af32'
);
assert.ok(qwenEmbedding, 'qwen-3-embedding-0-6b-q4k-ehf16-af32 must have an embedding compare profile');
assert.equal(qwenEmbedding.releaseClaimable, true);
assert.equal(qwenEmbedding.defaultDopplerSource, 'quickstart-registry');
assert.equal(qwenEmbedding.dopplerRuntimeProfile, 'profiles/qwen-3-embedding-0-6b-metal-throughput');
assert.equal(qwenEmbedding.dopplerVerifyRuntimeProfile, 'profiles/qwen-3-embedding-0-6b-metal-stability');
assert.equal(
  qwenEmbeddingMetalThroughputProfile.runtime.inference.compute,
  undefined,
  'Qwen embedding Metal throughput profile must not change manifest compute precision'
);
assert.equal(
  qwenEmbeddingMetalStabilityProfile.runtime.inference.compute,
  undefined,
  'Qwen embedding Metal stability profile must not change manifest compute precision'
);
assert.equal(
  qwenEmbeddingMetalThroughputProfile.runtime.inference.session.skipEmbeddingKVCacheWrites,
  true,
  'Qwen embedding Metal throughput profile must skip dead embedding KV cache writes'
);
assert.equal(
  qwenEmbeddingMetalStabilityProfile.runtime.inference.session.skipEmbeddingKVCacheWrites,
  true,
  'Qwen embedding Metal stability profile must skip dead embedding KV cache writes'
);

console.log('embedding-compare-contract.test: ok');
