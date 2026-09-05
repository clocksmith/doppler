import {
  SEQUENCE_REFERENCE_TRANSCRIPT_SCHEMA_ID,
  assertSequenceReferenceTranscript,
} from '../../config/sequence-reference.js';

export function buildSequenceReferenceTranscript(report, artifact, executionGraphHash) {
  if (report.schema !== 'doppler.sequenceModelQualification.v1' || report.passed !== true) {
    throw new Error('Program Bundle requires a passed sequence qualification report.');
  }
  if (report.runtime?.executionGraphHash !== executionGraphHash) {
    throw new Error('Sequence qualification execution graph does not match the Program Bundle.');
  }
  const checkpoint = report.model?.artifactIdentity?.sourceCheckpointId;
  const identity = report.result?.checks?.find((check) => check.id === 'model.identity');
  if (!checkpoint || checkpoint !== report.reference?.source?.checkpointId
    || identity?.actualCheckpointId !== checkpoint || identity?.expectedCheckpointId !== checkpoint
    || identity?.actualModelId !== report.model?.modelId || identity?.expectedModelId !== report.model?.modelId) {
    throw new Error('Sequence qualification source identity does not match its reference.');
  }
  const transcript = assertSequenceReferenceTranscript({
    schema: SEQUENCE_REFERENCE_TRANSCRIPT_SCHEMA_ID,
    operation: 'encodeSequence',
    modelId: report.model.modelId,
    surface: report.runtime.surface,
    executionGraphHash,
    manifestHash: report.model.manifestHash,
    source: { kind: 'sequence-qualification', path: artifact.path, hash: artifact.hash },
    reference: structuredClone(report.reference),
    options: structuredClone(report.result.options),
    output: {
      embeddingDim: report.result.embeddingDim,
      tokenCount: report.result.tokenCount,
      digests: structuredClone(report.result.outputDigests),
    },
    checks: structuredClone(report.result.checks),
  });
  return { artifact, transcript, adapter: {
    source: 'reference-report',
    surface: report.runtime.surface,
    deviceInfo: report.runtime.adapterInfo,
  } };
}
