import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {
  createReceiptFromFiles,
  materializeReleaseToJavaScriptReceipt,
  parseArgs,
} from '../../tools/create-release-to-javascript-receipt.js';
import { validateReleaseToJavaScriptReceipt } from '../../src/converter/release-to-javascript-receipt.js';

const repoRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-release-receipt-'));
await fs.mkdir(path.join(repoRoot, 'src'), { recursive: true });
await fs.mkdir(path.join(repoRoot, 'reports'), { recursive: true });
await fs.mkdir(path.join(repoRoot, '.tmp'), { recursive: true });
await fs.writeFile(path.join(repoRoot, 'src', 'accepted.js'), 'export const accepted = true;\n');
await fs.writeFile(path.join(repoRoot, 'reports', 'parity.json'), '{"status":"passed"}\n');
await fs.writeFile(path.join(repoRoot, '.tmp', 'pack.json'), '{"packId":"pack.test"}\n');

const spec = {
  campaignId: 'heterogeneous-model-ir-v2:qwen-test',
  source: {
    checkpointId: 'upstream/test',
    revision: 'abc123',
    publicationTimestampDisposition: 'unresolved',
    publishedAt: null,
  },
  startedAt: '2026-08-23T00:00:00.000Z',
  completedAt: '2026-08-23T01:00:00.000Z',
  humanInterventions: [
    { id: 'scope', kind: 'campaign-governance', actor: 'reviewer', disposition: 'accepted' },
  ],
  unresolvedFacts: [{
    id: 'source-publication-timestamp',
    disposition: 'unresolved',
    evidence: 'Pinned snapshot has no authoritative publication timestamp.',
  }],
  candidates: {
    generated: 2,
    rejected: [{ id: 'rejected' }],
    accepted: [{ id: 'accepted' }],
  },
  acceptedCode: { revision: 'deadbeef', files: ['src/accepted.js'] },
  qualification: { status: 'passed', packId: 'pack.test', packPath: '.tmp/pack.json' },
  evidence: [{ kind: 'parity', path: 'reports/parity.json' }],
};

const receipt = await materializeReleaseToJavaScriptReceipt(spec, { repoRoot });
assert.equal(validateReleaseToJavaScriptReceipt(receipt).ok, true);
assert.match(receipt.acceptedCode.files[0].digest, /^sha256:[0-9a-f]{64}$/);
assert.match(receipt.qualification.packDigest, /^sha256:[0-9a-f]{64}$/);
assert.match(receipt.evidence[0].digest, /^sha256:[0-9a-f]{64}$/);
assert.equal(receipt.humanAuthoredSemanticDecisions, 0);
assert.equal(receipt.elapsed.publicationToSignedPackMs, null);

const specPath = path.join(repoRoot, 'spec.json');
const outputPath = path.join(repoRoot, 'receipt.json');
await fs.writeFile(specPath, `${JSON.stringify(spec, null, 2)}\n`);
const fromFiles = await createReceiptFromFiles({ specPath, outputPath, repoRoot });
assert.deepEqual(JSON.parse(await fs.readFile(outputPath, 'utf8')), fromFiles.receipt);

await assert.rejects(
  materializeReleaseToJavaScriptReceipt({
    ...structuredClone(spec),
    acceptedCode: { revision: 'deadbeef', files: ['../outside.js'] },
  }, { repoRoot }),
  /inside the workspace/
);
assert.deepEqual(parseArgs(['--spec', 'spec.json', '--out', 'receipt.json', '--json']), {
  spec: 'spec.json', out: 'receipt.json', json: true,
});

await fs.rm(repoRoot, { recursive: true, force: true });
console.log('✔ create-release-to-javascript-receipt.test.js passed');
