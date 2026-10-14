import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { loadTrainingWorkloadCapsule } from '../../src/experimental/training/workloads.js';

// Per-capsule loading is covered by tools/verify-training-workload-capsules.js
// (training-workload-capsules.test.js). This test covers registry ID resolution.
const registry = JSON.parse(
  readFileSync('src/experimental/training/workload-capsules/registry.json', 'utf8')
);
for (const entry of registry.workloads) {
  const byId = await loadTrainingWorkloadCapsule(entry.id);
  const byPath = await loadTrainingWorkloadCapsule(entry.path);
  assert.equal(byId.absolutePath, byPath.absolutePath, `${entry.id}: ID must resolve to its registry path`);
  assert.equal(byId.workloadSha256, byPath.workloadSha256, `${entry.id}: ID and path loads must read the same bytes`);
  assert.equal(byId.workload.id, entry.id, `${entry.id}: loaded workload id must match its registry id`);
}

await assert.rejects(
  () => loadTrainingWorkloadCapsule('missing-training-workload-id'),
  /not found in registry/
);

console.log('training-workload-loader-v2.test: ok');
