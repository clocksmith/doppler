import { sha256Hex } from '../formats/sha256.js';
import { stableSortObject } from '../formats/stable-sort-object.js';
import { assertSequenceReferenceTranscript } from '../config/sequence-reference.js';
import { assertRerankReferenceTranscript, assertRerankSourceIdentity } from '../config/rerank-reference.js';
import { assertEmbeddingReferenceTranscript, assertEmbeddingSourceIdentity } from '../config/embedding-reference.js';
import { resolveCapsuleEmbeddingContract } from '../config/embedding-contract.js';

function isObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function hashStable(value) {
  return `sha256:${sha256Hex(JSON.stringify(stableSortObject(value)))}`;
}

export function promoteQualifiedModelIRV2(modelIR, programBundle) {
  const transcript = programBundle.referenceTranscript;
  const operation = transcript?.operation ?? 'generate';
  if (operation === 'rerank' || operation === 'embed') {
    if (operation === 'rerank') assertRerankReferenceTranscript(transcript);
    else assertEmbeddingReferenceTranscript(transcript);
    const source = transcript.reference.source;
    if (source.revision !== modelIR.sourceIdentity.revision
      || source.checkpointId !== modelIR.sourceIdentity.checkpointId
      || source.repository !== modelIR.sourceIdentity.repository) {
      throw new Error(`Forge ${operation} source identity does not match ModelIR source identity.`);
    }
  } else if (operation === 'generate') {
    const parity = transcript?.sourceParity;
    if (parity?.schema !== 'doppler.source-token-parity/v1' || parity.status !== 'passed'
      || parity.prompt?.passed !== true || parity.generation?.passed !== true) {
      throw new Error('Forge ModelIR v2 promotion requires exact passed source-token parity.');
    }
    if (parity.sourceRevision !== modelIR.sourceIdentity.revision
      || ![modelIR.sourceIdentity.checkpointId, modelIR.sourceIdentity.repository].includes(parity.sourceModel)) {
      throw new Error('Forge source-token parity identity does not match ModelIR source identity.');
    }
  } else {
    throw new Error(`Forge ModelIR v2 promotion does not support operation "${operation}".`);
  }
  const surface = transcript.surface;
  if (typeof surface !== 'string' || !surface.endsWith('-webgpu') || surface.startsWith('unknown')) {
    throw new Error('Forge ModelIR v2 promotion requires an explicit physical WebGPU surface.');
  }
  const entryPoints = modelIR.entryPoints.filter((entryPoint) => (
    entryPoint.kind === operation && entryPoint.status === 'lowered'
      && modelIR.supportScope.loweredEntryPoints.includes(entryPoint.id)
  ));
  if (entryPoints.length !== 1) {
    throw new Error(`Forge ModelIR v2 promotion requires exactly one lowered ${operation} entry point.`);
  }
  return {
    ...structuredClone(modelIR),
    supportScope: {
      ...structuredClone(modelIR.supportScope),
      qualifiedEntryPoints: [...new Set([
        ...modelIR.supportScope.qualifiedEntryPoints, entryPoints[0].id,
      ])].sort(),
    },
  };
}

export function buildQualificationRecords(lowered) {
  const normalized = lowered.normalized;
  const referenceArtifact = normalized.artifacts.find((artifact) => artifact.role === 'reference-report');
  if (!referenceArtifact) throw new Error('Forge requires a packaged reference-report artifact.');
  const transcript = normalized.programBundle.referenceTranscript;
  if (transcript?.operation === 'embed') {
    assertEmbeddingReferenceTranscript(transcript);
    assertEmbeddingSourceIdentity(normalized.manifest?.artifactIdentity, transcript.reference);
    if (hashStable(resolveCapsuleEmbeddingContract(normalized.manifest)) !== hashStable(transcript.reference.embeddingContract)
      || transcript.manifestHash !== normalized.manifestHash || transcript.modelId !== lowered.modelIR.modelId
      || transcript.executionGraphHash !== normalized.programBundle.execution.graphHash) {
      throw new Error('Forge embedding qualification does not match its declared embedding contract and exact program.');
    }
    const surfaces = normalized.programBundle.captureProfile?.surfaces;
    if (!Array.isArray(surfaces) || surfaces.length !== 1 || surfaces[0] !== transcript.surface) {
      throw new Error('Forge embedding capture surface must match the actual qualification report.');
    }
    return [{ surface: transcript.surface, status: 'passed', operation: 'embed',
      embeddedTexts: transcript.reference.input.texts.length,
      evidenceArtifactId: referenceArtifact.artifactId, evidenceHash: referenceArtifact.hash,
      transcriptHash: hashStable(transcript),
    }, ...normalized.qualificationEvidence.map(({ artifact, ...record }) => record)];
  }
  if (normalized.manifest?.modelType === 'embedding') {
    throw new Error('Forge requires text embedding qualification for an embedding model; other operation evidence is insufficient.');
  }
  if (transcript?.operation === 'rerank') {
    assertRerankReferenceTranscript(transcript);
    assertRerankSourceIdentity(normalized.manifest?.artifactIdentity, transcript.reference);
    if (normalized.manifest?.inference?.supportsRerank !== true
      || hashStable(normalized.manifest.inference.rerank) !== hashStable(transcript.reference.scoringConfig)
      || normalized.manifest.artifactIdentity?.sourceCheckpointId !== transcript.reference.source.checkpointId
      || transcript.manifestHash !== normalized.manifestHash || transcript.modelId !== lowered.modelIR.modelId
      || transcript.executionGraphHash !== normalized.programBundle.execution.graphHash) {
      throw new Error('Forge rerank qualification does not match its declared scoring contract and exact program.');
    }
    const surfaces = normalized.programBundle.captureProfile?.surfaces;
    if (!Array.isArray(surfaces) || surfaces.length !== 1 || surfaces[0] !== transcript.surface) {
      throw new Error('Forge rerank capture surface must match the actual qualification report.');
    }
    return [{ surface: transcript.surface, status: 'passed', operation: 'rerank',
      rerankedDocuments: transcript.reference.input.documents.length,
      evidenceArtifactId: referenceArtifact.artifactId, evidenceHash: referenceArtifact.hash,
      transcriptHash: hashStable(transcript),
    }, ...normalized.qualificationEvidence.map(({ artifact, ...record }) => record)];
  }
  if (normalized.manifest?.inference?.supportsRerank === true) {
    throw new Error('Forge requires rerank qualification for a reranker; generation evidence is insufficient.');
  }
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
