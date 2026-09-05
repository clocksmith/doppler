import assert from 'node:assert/strict';
import { assertRerankReferenceTranscript, evaluateRerankReference } from '../../src/config/rerank-reference.js';
import { buildRerankReferenceTranscript } from '../../src/tooling/program-bundle/rerank-reference.js';
import { buildQualificationRecords } from '../../src/converter/forge-qualification.js';
import { createRerankReferenceFixture } from '../helpers/rerank-reference-fixture.js';

const transcript = createRerankReferenceFixture();
assert.equal(assertRerankReferenceTranscript(transcript), transcript);
for (const change of [
  (v) => { v.observation.outputs[0].trueLogit += 1; },
  (v) => { v.observation.outputs[0].tokenIds[0] += 1; },
  (v) => { v.observation.outputs[0].score = NaN; },
  (v) => { v.observation.outputs[0].index = 1; },
  (v) => { v.observation.outputs.pop(); },
  (v) => { v.observation.input.documents[0] = 'changed'; },
  (v) => { v.observation.scoringConfig.score = 'logit_difference'; },
  (v) => { v.reference.tolerances.logitMaxAbs = 10; },
  (v) => { v.generationConfig = { temperature: 0 }; },
]) {
  const copy = structuredClone(transcript); change(copy);
  assert.throws(() => assertRerankReferenceTranscript(copy), /Invalid rerank reference/);
}
const failed = structuredClone(transcript.observation);
failed.outputs[0].trueLogit += 2;
assert.equal(evaluateRerankReference(transcript.reference, failed).passed, false);

const report = {
  schema: 'doppler.rerankModelQualification.v1', passed: true,
  model: { modelId: transcript.modelId, manifestHash: transcript.manifestHash,
    artifactIdentity: { sourceCheckpointId: transcript.reference.source.checkpointId,
      sourceRepo: transcript.reference.source.repository, sourceRevision: transcript.reference.source.revision } },
  runtime: { surface: transcript.surface, executionGraphHash: transcript.executionGraphHash },
  reference: transcript.reference, referenceDigest: transcript.referenceDigest, observation: transcript.observation,
};
const artifact = { artifactId: 'reference', role: 'reference-report', path: 'report.json', hash: transcript.source.hash };
const built = buildRerankReferenceTranscript(report, artifact, transcript.executionGraphHash);
assert.equal(built.transcript.operation, 'rerank');
report.observation = failed;
assert.throws(() => buildRerankReferenceTranscript(report, artifact, transcript.executionGraphHash), /source comparison failed/);
const lowered = {
  modelIR: { modelId: transcript.modelId },
  normalized: { artifacts: [artifact], manifestHash: transcript.manifestHash, qualificationEvidence: [],
    manifest: { artifactIdentity: report.model.artifactIdentity,
      inference: { supportsRerank: true, rerank: transcript.reference.scoringConfig } },
    programBundle: { referenceTranscript: transcript, execution: { graphHash: transcript.executionGraphHash },
      captureProfile: { surfaces: [transcript.surface] } } },
};
assert.equal(buildQualificationRecords(lowered)[0].rerankedDocuments, 2);
for (const field of ['sourceRepo', 'sourceRevision']) {
  const copy = structuredClone(lowered);
  delete copy.normalized.manifest.artifactIdentity[field];
  assert.throws(() => buildQualificationRecords(copy), /must bind the pinned source/);
}
for (const field of ['manifestHash', 'executionGraphHash', 'modelId']) {
  const copy = structuredClone(lowered);
  copy.normalized.programBundle.referenceTranscript[field] = field === 'modelId' ? 'another-model' : `sha256:${'2'.repeat(64)}`;
  assert.throws(() => buildQualificationRecords(copy), /does not match/);
}
lowered.normalized.programBundle.referenceTranscript = { tokens: { ids: [1] }, generationConfig: { temperature: 0 } };
assert.throws(() => buildQualificationRecords(lowered), /generation evidence is insufficient/);
console.log('rerank-reference.test: ok (synthetic qualification, no hardware claim)');
