import assert from 'node:assert/strict';

import { buildQuantizationContractArtifact } from '../../src/config/quantization-contract-check.js';

const artifact = buildQuantizationContractArtifact();

assert.equal(artifact.ok, true);
assert.ok(artifact.checks.every((entry) => entry.ok));
assert.equal(artifact.stats.sampledSizes, 520);

console.log('quantization-contract-check.test: ok');
