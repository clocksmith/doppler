import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { existsSync } from 'node:fs';
import crypto from 'node:crypto';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

import {
  buildBoundaryComparisonReceipt,
  compareFloat32Arrays,
  readNpyF32,
  selectCandidateBoundaryValues,
} from '../../tools/compare-source-boundaries.js';

const exact = compareFloat32Arrays(new Float32Array([1, 2]), new Float32Array([1, 2]));
assert.equal(exact.exact, true);
assert.equal(exact.cosineSimilarity, 1);
assert.throws(
  () => compareFloat32Arrays(new Float32Array([1]), new Float32Array([1, 2])),
  /element-count mismatch/
);
assert.deepEqual(
  Array.from(selectCandidateBoundaryValues(
    new Float32Array([0, 0]),
    new Float32Array([1, 2, 3, 4]),
    'last-row'
  )),
  [3, 4]
);

const policyPath = 'src/config/forge/reference/glimmer-30b-decode-boundary-comparison.json';
const policy = JSON.parse(await fs.readFile(policyPath, 'utf8'));
const expected = JSON.parse(await fs.readFile(policy.output, 'utf8'));
test('retained Glimmer boundary captures reproduce the historical rejected comparison', {
  skip: existsSync(policy.candidateRoot)
    ? false : `Local boundary capture unavailable: ${policy.candidateRoot}`,
}, async () => {
  const observed = await buildBoundaryComparisonReceipt(policyPath);
  assert.deepEqual(observed, expected);
  assert.equal(observed.capture.comparisonCount, 21);
  assert.equal(observed.finding.firstExactDivergence, 'generation.7.model.embedding.output');
  assert.equal(observed.finding.firstToleranceDivergence, null);
  assert.equal(observed.finding.classification, 'precision-lane-drift-within-diagnostic-tolerance');
  assert.equal(observed.finding.tokenParityPassed, false);
  assert.equal(observed.finding.promotionEligible, false);
});

// Synthetic bytes test comparison mechanics, never model or physical parity.
function npy(values) {
  const header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }";
  const headerLength = Math.ceil((10 + header.length + 1) / 16) * 16 - 10;
  const bytes = Buffer.alloc(10 + headerLength + values.length * 4);
  bytes.write('\x93NUMPY', 0, 'latin1');
  bytes[6] = 1;
  bytes.writeUInt16LE(headerLength, 8);
  bytes.write(`${header.padEnd(headerLength - 1)}\n`, 10, 'latin1');
  values.forEach((value, index) => bytes.writeFloatLE(value, 10 + headerLength + index * 4));
  return bytes;
}

test('boundary receipt checks bytes, numerical tolerance, parity and path custody', async () => {
  const fixtureRoot = await fs.mkdtemp(path.join(tmpdir(), 'doppler-boundary-fixture-'));
  const sourceBytes = npy([1, 2]);
  const digest = readNpyF32(sourceBytes).payloadDigest;
  assert.equal(digest, `sha256:${crypto.createHash('sha256').update(sourceBytes.subarray(-8)).digest('hex')}`);
  const fixturePolicy = {
    schema: policy.schema,
    sourceTranscript: 'source.json', candidateReport: 'candidate.json', candidateRoot: 'candidate',
    phase: 'decode', generationStep: 7, layers: [],
    globalMappings: [{ id: 'embedding', sourceBoundaryId: 'embedding', candidatePath: 'output.npy' }],
    tolerance: policy.tolerance,
    author: { kind: 'test', actor: 'synthetic-fixture' },
  };
  const source = {
    model: 'synthetic', revision: 'fixture', execution: { kind: 'synthetic' },
    boundaries: [{ boundaryId: 'embedding', phase: 'decode', generationStep: 7,
      artifact: { path: 'source.npy' }, fullTensorDigest: digest }],
  };
  const writeJson = (name, value) => fs.writeFile(path.join(fixtureRoot, name), JSON.stringify(value));
  const candidatePath = path.join(fixtureRoot, 'candidate/output.npy');
  try {
    await fs.mkdir(path.join(fixtureRoot, 'candidate'));
    await writeJson('policy.json', fixturePolicy);
    await writeJson('source.json', source);
    await writeJson('candidate.json', { modelId: 'synthetic', metrics: { sourceParity: { status: 'failed' } } });
    await fs.writeFile(path.join(fixtureRoot, 'source.npy'), sourceBytes);
    await fs.writeFile(candidatePath, npy([1, 2.01]));
    let receipt = await buildBoundaryComparisonReceipt('policy.json', fixtureRoot);
    assert.equal(receipt.capture.comparisonCount, 1);
    assert.equal(receipt.finding.firstExactDivergence, 'embedding');
    assert.equal(receipt.finding.firstToleranceDivergence, null);
    assert.equal(receipt.finding.promotionEligible, false);
    await fs.writeFile(candidatePath, npy([9, -3]));
    receipt = await buildBoundaryComparisonReceipt('policy.json', fixtureRoot);
    assert.equal(receipt.finding.firstToleranceDivergence, 'embedding');
    assert.equal(receipt.finding.promotionEligible, false);
    await fs.writeFile(candidatePath, sourceBytes);
    await writeJson('candidate.json', { metrics: { sourceParity: { status: 'passed' } } });
    assert.equal((await buildBoundaryComparisonReceipt('policy.json', fixtureRoot)).finding.promotionEligible, true);
    await fs.writeFile(path.join(fixtureRoot, 'source.npy'), npy([1, 3]));
    await assert.rejects(buildBoundaryComparisonReceipt('policy.json', fixtureRoot), /artifact digest mismatch/);
    await writeJson('source.json', { ...source, boundaries: [] });
    await assert.rejects(buildBoundaryComparisonReceipt('policy.json', fixtureRoot), /does not contain boundary/);
    await writeJson('policy.json', { ...fixturePolicy, sourceTranscript: '../outside.json' });
    await assert.rejects(buildBoundaryComparisonReceipt('policy.json', fixtureRoot), /must remain inside/);
  } finally {
    await fs.rm(fixtureRoot, { recursive: true, force: true });
  }
});

console.log('source-boundary-comparison.test: ok');
