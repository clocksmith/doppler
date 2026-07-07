import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

const config = JSON.parse(
  await fs.readFile(new URL('../../benchmarks/vendors/rerank-compare.config.json', import.meta.url), 'utf8')
);
const catalog = JSON.parse(
  await fs.readFile(new URL('../../models/catalog.json', import.meta.url), 'utf8')
);

const catalogByModelId = new Map(
  (Array.isArray(catalog?.models) ? catalog.models : [])
    .filter((entry) => typeof entry?.modelId === 'string' && entry.modelId.length > 0)
    .map((entry) => [entry.modelId, entry])
);

assert.equal(config.schemaVersion, 1);
assert.ok(Array.isArray(config.modelProfiles) && config.modelProfiles.length > 0);
assert.equal(typeof config.defaults?.query, 'string');
assert.ok(Array.isArray(config.defaults?.documents) && config.defaults.documents.length >= 2);
assert.equal(config.defaults?.expectedTopDocumentIndex, 0);

for (const profile of config.modelProfiles) {
  const modelId = profile?.dopplerModelId;
  assert.equal(typeof modelId, 'string', 'rerank compare profiles must define dopplerModelId');
  const catalogEntry = catalogByModelId.get(modelId);
  assert.ok(catalogEntry, `${modelId}: rerank compare profile must exist in models/catalog.json`);
  assert.equal(
    Array.isArray(catalogEntry.modes) && catalogEntry.modes.includes('rerank'),
    true,
    `${modelId}: rerank compare profiles must target rerank catalog entries`
  );
  assert.equal(profile.compareLane, 'performance_comparable', `${modelId}: rerank compare lane`);
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
  assert.equal(
    profile.defaultTjsModelId,
    catalogEntry.vendorBenchmark.transformersjs.repoId,
    `${modelId}: profile TJS repo must match catalog`
  );
  assert.equal(
    profile.defaultTjsDtype,
    catalogEntry.vendorBenchmark.transformersjs.dtype,
    `${modelId}: profile TJS dtype must match catalog`
  );

  if (profile.releaseClaimable === true) {
    assert.equal(
      profile.defaultDopplerSource,
      'quickstart-registry',
      `${modelId}: release-claimable rerank compares must measure a hosted artifact`
    );
    assert.equal(catalogEntry.lifecycle?.availability?.hf, true, `${modelId}: release-claimable rerank compares require hosted HF availability`);
    assert.equal(typeof catalogEntry.hf?.revision, 'string', `${modelId}: release-claimable rerank compares require pinned HF revision`);
    assert.equal(catalogEntry.artifactCompleteness, 'complete', `${modelId}: release-claimable rerank compares require complete artifacts`);
    assert.equal(catalogEntry.runtimePromotionState, 'manifest-owned', `${modelId}: release-claimable rerank compares require manifest-owned runtime metadata`);
    assert.equal(catalogEntry.weightsRefAllowed, false, `${modelId}: release-claimable rerank compares must not depend on weightsRef fallback`);
  }
}

const qwenReranker = config.modelProfiles.find(
  (entry) => entry.dopplerModelId === 'qwen-3-reranker-0-6b-q4k-ehf16-af32'
);
assert.ok(qwenReranker, 'qwen-3-reranker-0-6b-q4k-ehf16-af32 must have a rerank compare profile');
assert.equal(qwenReranker.releaseClaimable, true);
assert.equal(qwenReranker.defaultDopplerSource, 'quickstart-registry');
assert.equal(qwenReranker.defaultTjsDtype, 'q4');

console.log('rerank-compare-contract.test: ok');
