// Offline evidence projection. Never supplies expected logits or changes runtime output.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { gunzipSync } from 'node:zlib';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '../..');
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const captureBytes = gunzipSync(await readFile(resolve(root,
  'artifacts/paired-intermediates-20261012/release-0.6.28/browser-split/direction-0.json.gz')));
const capture = JSON.parse(captureBytes);
const controlBytes = await readFile(resolve(here, 'source-token-control.json'));
const sourceBytes = await readFile(resolve(here, 'source-token-result.json'));
const parityBytes = await readFile(resolve(here, 'source-token-parity.json'));
const control = JSON.parse(controlBytes);
const source = JSON.parse(sourceBytes);
const parity = JSON.parse(parityBytes);
const run = capture.runs[0];
assert.equal(sha(captureBytes), control.captureSha256);
assert.equal(capture.pairedBinding.archiveSha256, control.archiveSha256);
assert.equal(sha(controlBytes), parity.provenance.controlSha256);
assert.equal(sha(sourceBytes), parity.provenance.resultSha256);
assert.equal(source.generationControlSha256, sha(controlBytes));
assert.equal(source.modelIdentity, control.modelIdentity);
for (const row of capture.descriptors) assert.equal(row.descriptor.modelIdentity, control.modelIdentity);
assert.deepEqual(run.generation, source.settings);
assert.deepEqual(source.inputIds, control.result.tokenIds);
assert.equal(run.inputTokens, source.inputIds.length);
assert.equal(run.completed, true);
assert.equal(run.stopReason, 'eos-token');
assert.equal(source.stopReason, 'eos');
assert.deepEqual(run.cleanupErrors, []);
const tokens = run.numerical.map(row => row.tokenId);
assert.deepEqual(tokens, source.generatedTokenIds);
assert.deepEqual(tokens, parity.generation.runtimeTokens);
assert.equal(run.text, source.outputText);
for (const [index, row] of run.numerical.entries()) {
  assert.equal(row.matches, true);
  assert.equal(row.step, index);
  assert.deepEqual(row.inputTokenIds, index === 0 ? source.inputIds : [tokens[index - 1]]);
}
assert.equal(run.generation.temperature, 0);
assert.equal(run.generation.seed, undefined);
// Explicit null represents the observed greedy run's absent random seed.
const generationConfig = { ...run.generation, seed: null };
const sourceParity = {
  ...parity,
  expectedTranscriptPath: 'source-token-result.json',
  expectedTranscriptHash: `sha256:${sha(sourceBytes)}`,
  sampling: 'greedy',
  prompt: { ...parity.prompt, expectedCount: source.inputIds.length,
    observedCount: run.inputTokens, firstMismatchIndex: null },
  generation: { ...parity.generation, expectedCount: source.generatedTokenIds.length,
    observedCount: tokens.length, firstMismatchIndex: null },
};
// Full-vector accuracy is separately recomputed by produce-physical-reports.js.
const report = {
  schema: 'doppler.retained-split-generation-report/v1',
  scope: 'Observed first request in a two-physical-browser split run; short source-token parity only',
  surface: 'browser',
  output: run.text,
  deviceInfo: capture.partitionHosts,
  metrics: {
    promptInput: { tokenIds: source.inputIds },
    generationConfig,
    tokensGenerated: tokens.length,
    stopReason: run.stopReason,
    sourceParity,
    referenceTranscript: {
      surface: 'browser-webgpu',
      generationConfig,
      prompt: { tokenCount: source.inputIds.length },
      tokens: { ids: tokens, coverage: { mode: 'complete', omitted: 0 } },
      output: { tokensGenerated: tokens.length, stopReason: run.stopReason,
        stopTokenId: tokens.at(-1) },
    },
  },
  provenance: {
    captureSha256: sha(captureBytes),
    controlSha256: sha(controlBytes), sourceResultSha256: sha(sourceBytes),
    paritySha256: sha(parityBytes), archiveSha256: control.archiveSha256,
    rawCapture: 'artifacts/paired-intermediates-20261012/release-0.6.28/browser-split/direction-0.json.gz',
    independentAccuracy: 'Separate resident qualification reports compare all55 full-vocabulary vectors with the approved F64 target',
    applicationQualification: false, signedOpening: false, documentQuality: false,
  },
};
const destination = resolve(here, 'generation-report.json');
await writeFile(destination, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
console.log(JSON.stringify({ destination, captureSha256: sha(captureBytes), tokens }));
