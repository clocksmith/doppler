import { sha256Hex } from '../formats/sha256.js';
import { stableSortObject } from '../formats/stable-sort-object.js';
import { assertSequenceReferenceTranscript } from '../config/sequence-reference.js';

function isObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function hashStable(value) {
  return `sha256:${sha256Hex(JSON.stringify(stableSortObject(value)))}`;
}

export function buildQualificationRecords(lowered) {
  const normalized = lowered.normalized;
  const referenceArtifact = normalized.artifacts.find((artifact) => artifact.role === 'reference-report');
  if (!referenceArtifact) throw new Error('Forge requires a packaged reference-report artifact.');
  const transcript = normalized.programBundle.referenceTranscript;
  if (transcript?.operation === 'encodeSequence') {
    assertSequenceReferenceTranscript(transcript);
    if (lowered.modelIR.outputTopology?.headType !== 'sequence-encoder'
      || transcript.manifestHash !== normalized.manifestHash
      || transcript.modelId !== lowered.modelIR.modelId
      || transcript.executionGraphHash !== normalized.programBundle.execution.graphHash) {
      throw new Error('Forge sequence qualification does not match its encoder ModelIR and exact program.');
    }
    const surfaces = normalized.programBundle.captureProfile?.surfaces;
    if (!Array.isArray(surfaces) || surfaces.length !== 1 || surfaces[0] !== transcript.surface) {
      throw new Error('Forge sequence capture surface must match the actual qualification report.');
    }
    return [{
      surface: transcript.surface,
      status: 'passed',
      operation: 'encodeSequence',
      encodedSequences: 1,
      evidenceArtifactId: referenceArtifact.artifactId,
      evidenceHash: referenceArtifact.hash,
      transcriptHash: hashStable(transcript),
    }, ...normalized.qualificationEvidence.map(({ artifact, ...record }) => record)];
  }
  if (lowered.modelIR.outputTopology?.headType === 'sequence-encoder') {
    throw new Error('Forge requires sequence qualification for an encoder; generation evidence is insufficient.');
  }
  const tokens = transcript?.tokens?.ids;
  if (!Array.isArray(tokens) || tokens.length === 0) throw new Error('Forge requires reference transcript token IDs.');
  const generationConfig = transcript?.generationConfig;
  if (!isObject(generationConfig) || !Number.isFinite(generationConfig.temperature)) {
    throw new Error('Forge requires reference transcript generationConfig.');
  }
  if (generationConfig.temperature > 0 && !Number.isFinite(generationConfig.seed)) {
    throw new Error('Forge rejects nondeterministic qualification evidence without a seed.');
  }
  const surfaces = normalized.programBundle.captureProfile?.surfaces;
  if (!Array.isArray(surfaces) || surfaces.length === 0) throw new Error('Forge requires captureProfile.surfaces qualification evidence.');
  const records = surfaces.map((surface) => ({
    surface,
    status: 'passed',
    evidenceArtifactId: referenceArtifact.artifactId,
    evidenceHash: referenceArtifact.hash,
    transcriptHash: hashStable({ surface, captureProfile: normalized.programBundle.captureProfile, transcript }),
    generatedTokens: tokens.length,
  }));
  for (const evidence of normalized.qualificationEvidence) {
    const { artifact: ignoredArtifact, ...record } = evidence;
    void ignoredArtifact;
    records.push(record);
  }
  return records;
}
