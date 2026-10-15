import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const catalog = JSON.parse(await readFile('models/catalog.json', 'utf8'));
const scoreboard = JSON.parse(
  await readFile('benchmarks/vendors/model-competition-scoreboard.json', 'utf8')
);
const catalogByModelId = new Map(
  catalog.models.map((model) => [model.modelId, model])
);

for (const row of scoreboard.rows) {
  const model = catalogByModelId.get(row.modelId);
  assert.ok(model, `${row.rowId}: scoreboard model must exist in the catalog`);
  if (row.metrics != null) {
    assert.equal(
      model.lifecycle?.status?.tested,
      'verified',
      `${row.rowId}: benchmark metrics require a currently verified catalog artifact`
    );
    assert.equal(
      model.lifecycle?.tested?.result,
      'pass',
      `${row.rowId}: benchmark metrics require a currently passing catalog artifact`
    );
    if (row.workloadMode === 'generation' && row.evidence?.compareResult) {
      const receipt = JSON.parse(await readFile(row.evidence.compareResult, 'utf8'));
      const expected = model.vendorBenchmark?.transformersjs?.repoId;
      if (receipt.tjsModelId && expected && receipt.tjsModelId !== expected) {
        assert.equal(row.competitor.modelId, receipt.tjsModelId,
          `${row.rowId}: historical measurements must retain the measured comparator`);
        assert.equal(row.claimStatus, 'superseded-baseline',
          `${row.rowId}: obsolete comparator evidence cannot become a current claim`);
        assert.ok(row.missing.includes('current-comparator-result'));
      }
    }
  }
  if (model.lifecycle?.status?.tested === 'failed') {
    assert.equal(row.metrics, null, `${row.rowId}: failed artifacts cannot retain scoreboard metrics`);
    assert.ok(
      Object.values(row.platforms).every((status) => status === 'missing'),
      `${row.rowId}: failed artifacts cannot retain verified or benchmarked platform labels`
    );
  }
}

console.log('model-competition-lifecycle-contract.test: ok');
