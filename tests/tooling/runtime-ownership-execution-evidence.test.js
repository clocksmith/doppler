import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import {
  hashRuntimeOwnershipExecutionEvidence,
  parseArgs,
} from '../../tools/hash-runtime-ownership-execution-evidence.js';
import {
  computeRuntimeOwnershipEvidenceId,
  validateRuntimeOwnershipExecutionEvidence,
} from '../../tools/lib/runtime-ownership-execution-evidence.js';

const SHA_A = `sha256:${'a'.repeat(64)}`;
const SHA_B = `sha256:${'b'.repeat(64)}`;
const SHA_C = `sha256:${'c'.repeat(64)}`;

function receipt(overrides = {}) {
  return {
    schema: 'doppler.runtime-ownership-execution-evidence/v1',
    role: 'source',
    providerId: 'authoritative-source-runtime',
    artifactId: 'fixture/source-model',
    artifactRevision: '0123456789abcdef',
    workload: 'generation',
    logicalModelId: 'fixture-logical-model',
    runtime: {
      name: 'fixture-runtime',
      version: '1.0.0',
      backendId: 'fixture-backend',
      environmentFingerprint: SHA_A,
    },
    invocation: { configurationDigest: SHA_B },
    result: {
      status: 'passed',
      outputDigest: SHA_C,
      startedAtUtc: '2026-08-15T10:00:00.000Z',
      completedAtUtc: '2026-08-15T10:01:00.000Z',
    },
    ...overrides,
  };
}

{
  const value = receipt();
  const result = validateRuntimeOwnershipExecutionEvidence(value, {
    role: 'source',
    providerId: value.providerId,
    artifactId: value.artifactId,
    workload: value.workload,
    logicalModelId: value.logicalModelId,
  });
  assert.deepEqual(result.errors, []);
  assert.deepEqual(result.reasons, []);
  assert.match(result.evidenceId, /^sha256:[0-9a-f]{64}$/);
  const reordered = Object.fromEntries(Object.entries(value).reverse());
  assert.equal(computeRuntimeOwnershipEvidenceId(reordered), result.evidenceId);
}

{
  const result = validateRuntimeOwnershipExecutionEvidence(receipt(), {
    providerId: 'different-provider',
  });
  assert.ok(result.errors.some((error) => error.includes('does not match expected')));
  assert.equal(result.evidenceId, null);
}

{
  const value = receipt({
    result: {
      status: 'failed',
      outputDigest: null,
      startedAtUtc: '2026-08-15T10:00:00.000Z',
      completedAtUtc: '2026-08-15T10:01:00.000Z',
    },
  });
  const result = validateRuntimeOwnershipExecutionEvidence(value);
  assert.deepEqual(result.errors, []);
  assert.deepEqual(result.reasons, ['source-execution-not-passed']);
  assert.match(result.evidenceId, /^sha256:[0-9a-f]{64}$/);
}

{
  const testRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-ownership-evidence-'));
  try {
    const receiptPath = path.join(testRoot, 'receipt.json');
    const value = receipt();
    await fs.writeFile(receiptPath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
    const result = await hashRuntimeOwnershipExecutionEvidence(receiptPath);
    assert.equal(result.executionId, computeRuntimeOwnershipEvidenceId(value));
    assert.equal(result.status, 'passed');
  } finally {
    await fs.rm(testRoot, { recursive: true, force: true });
  }
}

assert.deepEqual(parseArgs(['--receipt', 'receipt.json', '--json']), {
  receiptPath: 'receipt.json',
  json: true,
});
assert.throws(() => parseArgs([]), /--receipt is required/);

console.log('runtime-ownership-execution-evidence.test: ok');
