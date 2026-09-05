import { RERANK_REFERENCE_TRANSCRIPT_SCHEMA_ID, assertRerankReferenceTranscript, assertRerankSourceIdentity } from '../../config/rerank-reference.js';

export function buildRerankReferenceTranscript(report, artifact, executionGraphHash) {
  if (report.schema !== 'doppler.rerankModelQualification.v1' || report.passed !== true
    || report.runtime?.executionGraphHash !== executionGraphHash) {
    throw new Error('Program Bundle requires passed rerank qualification for its exact execution graph.');
  }
  if (report.model?.artifactIdentity?.sourceCheckpointId !== report.reference?.source?.checkpointId) {
    throw new Error('Rerank qualification checkpoint differs from its source reference.');
  }
  const transcript = assertRerankReferenceTranscript({
    schema: RERANK_REFERENCE_TRANSCRIPT_SCHEMA_ID, operation: 'rerank',
    modelId: report.model.modelId, surface: report.runtime.surface,
    executionGraphHash, manifestHash: report.model.manifestHash,
    source: { kind: 'rerank-qualification', path: artifact.path, hash: artifact.hash },
    reference: structuredClone(report.reference), referenceDigest: report.referenceDigest,
    observation: structuredClone(report.observation),
  });
  assertRerankSourceIdentity(report.model.artifactIdentity, transcript.reference);
  return { artifact, transcript, adapter: { source: 'reference-report',
    surface: report.runtime.surface, deviceInfo: report.runtime.adapterInfo } };
}
