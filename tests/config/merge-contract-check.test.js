import assert from 'node:assert/strict';

import { buildMergeContractArtifact } from '../../src/config/merge-contract-check.js';
import { createDopplerConfig } from '../../src/config/schema/doppler.schema.js';

const artifact = buildMergeContractArtifact();

assert.equal(artifact.schemaVersion, 1);
assert.equal(artifact.ok, true);
assert.ok(artifact.checks.length >= 8);
assert.equal(
  artifact.checks.some((entry) => entry.id === 'runtime.mergeConfig.defined_overlay_preserves_null' && entry.ok),
  true
);
assert.equal(
  artifact.checks.some((entry) => entry.id === 'runtime.mergeConfig.pipeline_preserves_manifest_value' && entry.ok),
  true
);
assert.equal(
  artifact.checks.some((entry) => entry.id === 'runtime.schema.defaults_are_isolated_per_instance' && entry.ok && entry.mode === 'actual'),
  true
);
assert.equal(
  artifact.checks.some((entry) => entry.id === 'runtime.schema.calibrate_does_not_mutate_kernel_warmup_defaults' && entry.ok && entry.mode === 'actual'),
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
assert.equal(
  artifact.checks.some((entry) => entry.id === 'runtime.mergeShallowObject.invalid_explicit_override_fails_closed' && entry.ok && entry.mode === 'actual'),
  true
);

assert.throws(
  () => createDopplerConfig({
    runtime: {
      inference: {
        kernelPathPolicy: {
          sourceScope: ['runtime'],
        },
      },
    },
  }),
  /does not accept legacy "runtime". Use "config"/
);

assert.throws(
  () => createDopplerConfig({
    runtime: {
      inference: {
        kernelPathPolicy: ['invalid'],
      },
    },
  }),
  /kernelPathPolicy must be an object/
);

assert.throws(
  () => createDopplerConfig({
    runtime: {
      inference: {
        chatTemplate: null,
      },
    },
  }),
  /shallow object overrides must be plain objects/
);

console.log('merge-contract-check.test: ok');
