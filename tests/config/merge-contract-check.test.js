import assert from 'node:assert/strict';

import { buildMergeContractArtifact } from '../../src/config/merge-contract-check.js';

const artifact = buildMergeContractArtifact();

assert.equal(artifact.schemaVersion, 1);
assert.equal(artifact.ok, true);
assert.ok(artifact.checks.length >= 6);
assert.equal(
  artifact.checks.some((entry) => entry.id === 'loader.architecture.nullish_null_falls_through' && entry.ok),
  true
);
assert.equal(
  artifact.checks.some((entry) => entry.id === 'runtime.mergeConfig.defined_overlay_preserves_null' && entry.ok),
  true
);
assert.equal(
  artifact.checks.some((entry) => entry.id === 'runtime.inference.session.subtree_override_replaces_base' && entry.ok && entry.mode === 'actual'),
  true
);
assert.equal(
  artifact.checks.some((entry) => entry.id === 'runtime.kernelPathPolicy.source_scope_mirrors_allow_sources' && entry.ok && entry.mode === 'actual'),
  true
);
assert.equal(
  artifact.checks.some((entry) => entry.id === 'runtime.schema.kernelPathPolicy.helper_is_used' && entry.ok && entry.mode === 'actual'),
  true
);
assert.equal(
  artifact.checks.some((entry) => entry.id === 'runtime.mergeHelpers.chooseDefinedWithSource.runtime_marks_source' && entry.ok && entry.mode === 'actual'),
  true
);
assert.equal(
  artifact.checks.some((entry) => entry.id === 'runtime.mergeHelpers.chooseDefinedWithSource.manifest_marks_source' && entry.ok && entry.mode === 'actual'),
  true
);

console.log('merge-contract-check.test: ok');
