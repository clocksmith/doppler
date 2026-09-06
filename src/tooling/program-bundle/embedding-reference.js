import { EMBEDDING_REFERENCE_TRANSCRIPT_SCHEMA_ID, assertEmbeddingReferenceTranscript, assertEmbeddingSourceIdentity } from '../../config/embedding-reference.js';

export function buildEmbeddingReferenceTranscript(report, artifact, executionGraphHash) {
  if (report.schema !== 'doppler.embeddingModelQualification.v1' || report.passed !== true
    || report.runtime?.executionGraphHash !== executionGraphHash) {
    throw new Error('Program Bundle requires passed embedding qualification for its exact execution graph.');
  }
  const transcript = assertEmbeddingReferenceTranscript({
    schema: EMBEDDING_REFERENCE_TRANSCRIPT_SCHEMA_ID, operation: 'embed',
    modelId: report.model.modelId, surface: report.runtime.surface,
    executionGraphHash, manifestHash: report.model.manifestHash,
    source: { kind: 'embedding-qualification', path: artifact.path, hash: artifact.hash },
    reference: structuredClone(report.reference), referenceDigest: report.referenceDigest,
    observation: structuredClone(report.observation),
  });
  assertEmbeddingSourceIdentity(report.model.artifactIdentity, transcript.reference);
  return { artifact, transcript, adapter: { source: 'reference-report',
    surface: report.runtime.surface, deviceInfo: report.runtime.adapterInfo } };
}
