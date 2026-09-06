import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { buildReferenceTranscript } from '../../src/tooling/program-bundle/materialize.js';
import { buildEmbeddingReferenceTranscript } from '../../src/tooling/program-bundle/embedding-reference.js';
import { createEmbeddingReferenceFixture } from '../helpers/embedding-reference-fixture.js';

const fixture = createEmbeddingReferenceFixture();
const report = {
  schema: 'doppler.embeddingModelQualification.v1', passed: true,
  model: { modelId: fixture.modelId, manifestHash: fixture.manifestHash, artifactIdentity: {
    sourceCheckpointId: fixture.reference.source.checkpointId,
    sourceRepo: fixture.reference.source.repository, sourceRevision: fixture.reference.source.revision,
  } },
  runtime: { surface: fixture.surface, executionGraphHash: fixture.executionGraphHash, adapterInfo: { vendor: 'synthetic' } },
  reference: fixture.reference, referenceDigest: fixture.referenceDigest, observation: fixture.observation,
};
const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-embed-reference-'));
try {
  const reportPath = path.join(directory, 'report.json');
  await fs.writeFile(reportPath, JSON.stringify(report));
  const result = await buildReferenceTranscript(reportPath, directory, fixture.executionGraphHash);
  assert.equal(result.transcript.operation, 'embed');
  assert.equal(result.transcript.referenceDigest, fixture.referenceDigest);
  assert.equal(result.transcript.source.hash, result.artifact.hash);
  assert.equal(result.adapter.surface, fixture.surface);
  for (const change of [
    value => { value.passed = false; },
    value => { value.runtime.executionGraphHash = `sha256:${'2'.repeat(64)}`; },
    value => { value.model.artifactIdentity.sourceRevision = '2'.repeat(40); },
    value => { value.observation.outputs[0].embedding[0] = 0.5; },
  ]) {
    const invalid = structuredClone(report);
    change(invalid);
    assert.throws(() => buildEmbeddingReferenceTranscript(invalid, result.artifact, fixture.executionGraphHash));
  }
} finally { await fs.rm(directory, { recursive: true, force: true }); }
console.log('embedding-reference-materialization.test: passed (synthetic references)');
