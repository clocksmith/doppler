import assert from 'node:assert/strict';

import {
  buildKernelPathContractArtifact,
  extractKernelPathContractFacts,
  validateKernelPathContractFacts,
} from '../../src/config/kernel-path-contract-check.js';
import { getKernelPathContractArtifact } from '../../src/config/kernel-path-loader.js';

{
  const artifact = getKernelPathContractArtifact();
  assert.equal(artifact.ok, true);
  assert.equal(artifact.schemaVersion, 1);
  assert.ok(artifact.stats.totalEntries > 0);
  assert.ok(artifact.stats.fallbackMappings > 0);
  assert.deepEqual(
    artifact.checks.map((entry) => entry.ok),
    [true, true, true, true]
  );
}

{
  const facts = extractKernelPathContractFacts({
    registryId: 'synthetic-kernel-paths',
    entries: [
      { id: 'cycle-a', aliasOf: 'cycle-b' },
      { id: 'cycle-b', aliasOf: 'cycle-a' },
      { id: 'missing-alias', aliasOf: 'nope' },
      { id: 'source-path', file: 'source-path.json' },
      { id: 'narrow-path', file: 'narrow-path.json' },
    ],
    fallbackMappings: [
      {
        primaryKernelPathId: 'source-path',
        fallbackKernelPathId: 'narrow-path',
        primaryActivationDtype: 'f32',
        fallbackActivationDtype: 'f16',
      },
    ],
  });

  const result = validateKernelPathContractFacts(facts);
  assert.equal(result.ok, false);
  assert.ok(
    result.errors.some((message) => message.includes('aliases missing target "nope"'))
  );
  assert.ok(
    result.errors.some((message) => message.includes('alias cycle detected: cycle-a -> cycle-b -> cycle-a'))
  );
  assert.ok(
    result.errors.some((message) => message.includes('narrows activation dtype f32 -> f16'))
  );
  assert.deepEqual(
    result.checks.map((entry) => entry.ok),
    [false, false, true, false]
  );
}

{
  const artifact = buildKernelPathContractArtifact({
    registryId: 'simple-kernel-paths',
    entries: [
      { id: 'primary', file: 'primary.json' },
      { id: 'fallback', file: 'fallback.json' },
    ],
    fallbackMappings: [
      {
        primaryKernelPathId: 'primary',
        fallbackKernelPathId: 'fallback',
        primaryActivationDtype: 'f16',
        fallbackActivationDtype: 'f32',
      },
    ],
  });

  assert.equal(artifact.ok, true);
  assert.equal(artifact.stats.aliasEntries, 0);
  assert.equal(artifact.stats.fallbackMappings, 1);
}

console.log('kernel-path-contract-check.test: ok');
