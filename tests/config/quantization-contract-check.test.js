import assert from 'node:assert/strict';

import { buildQuantizationContractArtifact } from '../../src/config/quantization-contract-check.js';

const artifact = buildQuantizationContractArtifact();

// Structural assertions
assert.equal(typeof artifact, 'object');
assert.equal(artifact.schemaVersion, 1);
assert.equal(artifact.source, 'doppler');

// All checks pass
assert.equal(artifact.ok, true);
assert.ok(Array.isArray(artifact.checks));
assert.ok(artifact.checks.length >= 4, `expected at least 4 checks, got ${artifact.checks.length}`);
assert.ok(artifact.checks.every((entry) => entry.ok));

// Each check has required shape
for (const check of artifact.checks) {
  assert.equal(typeof check.id, 'string');
  assert.ok(check.id.length > 0, 'check.id must be non-empty');
  assert.equal(typeof check.ok, 'boolean');
}

// No errors when ok
assert.ok(Array.isArray(artifact.errors));
assert.equal(artifact.errors.length, 0);

// Stats
assert.equal(artifact.stats.sampledSizes, 520);

// Verify specific check IDs are present
const checkIds = artifact.checks.map((c) => c.id);
assert.ok(checkIds.includes('quantization.constants.schema'), 'missing schema check');
assert.ok(checkIds.includes('quantization.constants.crossModule'), 'missing crossModule check');
assert.ok(checkIds.includes('quantization.padToQ4KBlock.properties'), 'missing padToQ4KBlock check');
assert.ok(checkIds.includes('quantization.q4kBlockCount.coverage'), 'missing q4kBlockCount check');

console.log('quantization-contract-check.test: ok');
