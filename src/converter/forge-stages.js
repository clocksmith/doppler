/**
 * Doppler Forge's ten ahead-of-time compiler stages.
 *
 * Every semantic value in these stages comes from a manifest, a closed Program
 * Bundle, or qualification evidence. Unknown source facts are compile errors.
 *
 * @module converter/forge-stages
 */

import path from 'node:path';
import { createModelIR, hashModelIR, validateModelIR } from '../config/model-ir.js';
import {
  createTargetPlan,
  createTargetPlanV2,
  hashTargetPlan,
  validateTargetPlan,
} from '../config/target-plan.js';
import { validateInitialExecutionIdentity } from '../config/initial-execution-identity.js';
import {
  PACK_V2_PROGRAM_SCHEMA_ID,
  buildPackV2,
  signPackV2,
} from '../config/pack-v2.js';
import { sha256Hex } from '../utils/sha256.js';
import { stableSortObject } from '../utils/stable-sort-object.js';

export const FORGE_PIPELINE_VERSION = '2.0.0';

function isObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function requireObject(value, label) {
  if (!isObject(value)) throw new Error(`Forge requires ${label} as an object.`);
  return value;
}

function requireString(value, label) {
  if (typeof value !== 'string' || !value.trim()) throw new Error(`Forge requires ${label}.`);
  return value.trim();
}

function requirePositiveInteger(value, label) {
  if (!Number.isInteger(value) || value < 1) throw new Error(`Forge requires ${label} as a positive integer.`);
  return value;
}

function hashStable(value) {
  return `sha256:${sha256Hex(JSON.stringify(stableSortObject(value)))}`;
}

function toPosix(value) {
  return value.split(path.sep).join('/');
}

function artifactId(role, artifactPath, hash) {
  const name = path.basename(artifactPath).replace(/[^a-zA-Z0-9._-]+/g, '-');
  return `${role}:${name}:${hash.slice('sha256:'.length, 'sha256:'.length + 12)}`;
}

function resolveLayerType(layerIndex, layerPattern) {
  if (layerPattern.type === 'every_n') {
    const period = requirePositiveInteger(layerPattern.period, 'manifest.inference.layerPattern.period');
    if (!Number.isInteger(layerPattern.offset) || layerPattern.offset < 0 || layerPattern.offset >= period) {
      throw new Error('Forge requires manifest.inference.layerPattern.offset within the declared period.');
    }
    return layerIndex % period === layerPattern.offset ? 'global-attention' : 'local-attention';
  }
  if (layerPattern.type === 'explicit') {
    return requireString(layerPattern.layerTypes?.[layerIndex], `manifest.inference.layerPattern.layerTypes[${layerIndex}]`);
  }
  throw new Error(`Forge does not support layerPattern.type "${layerPattern.type}" without an explicit lowering.`);
}

function resolveArtifactSourcePath(artifact, input) {
  const bundleRoot = path.dirname(input.programBundlePath);
  if (artifact.role === 'wgsl-source' || artifact.role === 'host-source') {
    return path.resolve(bundleRoot, artifact.path);
  }
  return path.resolve(input.repoRoot, artifact.path);
}

function resolveLogicalArtifactPath(artifact) {
  const name = path.basename(artifact.path);
  if (artifact.role === 'wgsl-source' || artifact.role === 'host-source') {
    return toPosix(path.join('artifacts', artifact.path));
  }
  if (artifact.role === 'manifest' || artifact.role === 'weight-shard' || artifact.role === 'tokenizer') {
    return toPosix(path.join('artifacts', 'model', name));
  }
  if (artifact.role === 'reference-report' || artifact.role === 'qualification-evidence') {
    return toPosix(path.join('artifacts', 'evidence', name));
  }
  return toPosix(path.join('artifacts', 'source', name));
}

function assertModelTopologyRepresentable(manifest) {
  const architecture = manifest?.architecture ?? {};
  const inference = manifest?.inference ?? {};
  const candidates = [
    ['manifest.components', manifest?.components],
    ['manifest.visionEncoder', manifest?.visionEncoder],
    ['manifest.perceptionEncoder', manifest?.perceptionEncoder],
    ['manifest.speculativeDrafter', manifest?.speculativeDrafter],
    ['manifest.architecture.blockTypes', architecture.blockTypes],
    ['manifest.architecture.blockTypePattern', architecture.blockTypePattern],
    ['manifest.architecture.attentionGeometries', architecture.attentionGeometries],
    ['manifest.architecture.linearAttention', architecture.linearAttention],
    ['manifest.architecture.recurrentState', architecture.recurrentState],
    ['manifest.inference.blockTopology', inference.blockTopology],
    ['manifest.inference.linearAttention', inference.linearAttention],
    ['manifest.inference.recurrentState', inference.recurrentState],
    ['manifest.inference.multiTokenPrediction', inference.multiTokenPrediction],
    ['manifest.inference.visionEncoder', inference.visionEncoder],
    ['manifest.inference.perceptionEncoder', inference.perceptionEncoder],
    ['manifest.inference.speculativeDrafter', inference.speculativeDrafter],
  ];
  const unsupported = candidates.filter(([, value]) => (
    value !== undefined && value !== null && value !== false
  ));
  if (unsupported.length > 0) {
    throw new Error(
      `Forge ModelIR v1 cannot represent source topology: ${unsupported.map(([field]) => field).join(', ')}.`
    );
  }
}

function promoteQualifiedModelIRV2(modelIR, programBundle) {
  const parity = programBundle.referenceTranscript?.sourceParity;
  if (parity?.schema !== 'doppler.source-token-parity/v1' || parity.status !== 'passed'
    || parity.prompt?.passed !== true || parity.generation?.passed !== true) {
    throw new Error('Forge ModelIR v2 promotion requires exact passed source-token parity.');
  }
  if (parity.sourceRevision !== modelIR.sourceIdentity.revision
    || ![modelIR.sourceIdentity.checkpointId, modelIR.sourceIdentity.repository].includes(parity.sourceModel)) {
    throw new Error('Forge source-token parity identity does not match ModelIR source identity.');
  }
  const surface = programBundle.referenceTranscript?.surface;
  if (typeof surface !== 'string' || !surface.endsWith('-webgpu') || surface.startsWith('unknown')) {
    throw new Error('Forge ModelIR v2 promotion requires an explicit physical WebGPU surface.');
  }
  const entryPoints = modelIR.entryPoints.filter((entryPoint) => (
    entryPoint.kind === 'generate'
      && entryPoint.status === 'lowered'
      && modelIR.supportScope.loweredEntryPoints.includes(entryPoint.id)
  ));
  if (entryPoints.length !== 1) {
    throw new Error('Forge ModelIR v2 promotion requires exactly one lowered generate entry point.');
  }
  return {
    ...structuredClone(modelIR),
    supportScope: {
      ...structuredClone(modelIR.supportScope),
      qualifiedEntryPoints: [...new Set([
        ...modelIR.supportScope.qualifiedEntryPoints,
        entryPoints[0].id,
      ])].sort(),
    },
  };
}

function normalizePackArtifact(artifact, input) {
  const sourcePath = resolveArtifactSourcePath(artifact, input);
  const packPath = resolveLogicalArtifactPath(artifact);
  return {
    artifactId: artifactId(artifact.role, packPath, artifact.hash),
    role: artifact.role,
    path: packPath,
    hash: artifact.hash,
    sizeBytes: artifact.sizeBytes,
    sourcePath,
  };
}

function normalizeQualificationEvidence(evidence) {
  requireObject(evidence, 'qualification evidence');
  const surface = requireString(evidence.surface, 'qualificationEvidence.surface');
  const evidenceHash = requireString(evidence.evidenceHash, 'qualificationEvidence.evidenceHash');
  const sourcePath = path.resolve(requireString(evidence.sourcePath, 'qualificationEvidence.sourcePath'));
  const sizeBytes = requirePositiveInteger(evidence.sizeBytes, 'qualificationEvidence.sizeBytes');
  const generatedTokens = requirePositiveInteger(evidence.generatedTokens, 'qualificationEvidence.generatedTokens');
  const transcriptHash = requireString(evidence.transcriptHash, 'qualificationEvidence.transcriptHash');
  if (evidence.status !== 'passed') throw new Error('Forge only packages passed qualification evidence.');
  const packPath = toPosix(path.join(
    'artifacts',
    'evidence',
    `qualification-${surface.replace(/[^a-zA-Z0-9._-]+/g, '-')}-${evidenceHash.slice('sha256:'.length, 'sha256:'.length + 12)}.json`
  ));
  const artifact = {
    artifactId: artifactId('qualification-evidence', packPath, evidenceHash),
    role: 'qualification-evidence',
    path: packPath,
    hash: evidenceHash,
    sizeBytes,
    sourcePath,
  };
  return {
    surface,
    status: 'passed',
    evidenceArtifactId: artifact.artifactId,
    evidenceHash,
    transcriptHash,
    generatedTokens,
    artifact,
  };
}

function stripForgeOnlyArtifactFields(artifact) {
  const { sourcePath: ignoredSourcePath, ...packArtifact } = artifact;
  void ignoredSourcePath;
  return packArtifact;
}

/** Stage 1: Inspect pinned source objects and identities. */
export async function stageInspect(input) {
  requireObject(input, 'inspect input');
  const manifest = requireObject(input.manifest, 'manifest');
  const programBundle = requireObject(input.programBundle, 'Program Bundle');
  if (typeof input.manifestRaw !== 'string' || input.manifestRaw.length === 0) {
    throw new Error('Forge requires raw manifest bytes.');
  }
  if (typeof input.programBundleRaw !== 'string' || input.programBundleRaw.length === 0) {
    throw new Error('Forge requires raw Program Bundle bytes.');
  }
  const manifestRaw = input.manifestRaw;
  const programBundleRaw = input.programBundleRaw;
  const repoRoot = path.resolve(requireString(input.repoRoot, 'repoRoot'));
  const programBundlePath = path.resolve(requireString(input.programBundlePath, 'programBundlePath'));
  const outputPath = path.resolve(requireString(input.outputPath, 'outputPath'));
  if (manifest.modelId !== programBundle.modelId) {
    throw new Error(`Forge source mismatch: manifest modelId "${manifest.modelId}" != Program Bundle modelId "${programBundle.modelId}".`);
  }
  return {
    stage: 'inspect',
    ok: true,
    data: {
      manifest,
      manifestRaw,
      programBundle,
      programBundleRaw,
      repoRoot,
      programBundlePath,
      outputPath,
      qualificationEvidence: Array.isArray(input.qualificationEvidence) ? input.qualificationEvidence : [],
      modelIR: input.modelIR ?? null,
      modelIREvidence: input.modelIREvidence ?? null,
      initialExecutionIdentity: input.initialExecutionIdentity ?? null,
    },
  };
}

/** Stage 2: Normalize source identities and Pack-relative artifact locations. */
export function stageNormalize(inspected) {
  const input = requireObject(inspected, 'inspect output');
  const bundle = input.programBundle;
  const manifestHash = `sha256:${sha256Hex(input.manifestRaw)}`;
  if (bundle.sources?.manifest?.hash !== manifestHash) {
    throw new Error(`Forge manifest bytes do not match Program Bundle: expected ${bundle.sources?.manifest?.hash}, got ${manifestHash}.`);
  }
  if (bundle.execution?.graphHash !== bundle.sources?.executionGraph?.hash) {
    throw new Error('Forge Program Bundle execution graph identities disagree.');
  }
  const artifacts = bundle.artifacts.map((artifact) => normalizePackArtifact(artifact, input));
  const qualificationEvidence = input.qualificationEvidence.map(normalizeQualificationEvidence);
  artifacts.push(...qualificationEvidence.map((evidence) => evidence.artifact));
  let modelIREvidenceArtifactId = null;
  if (input.modelIREvidence !== null) {
    const evidence = requireObject(input.modelIREvidence, 'ModelIR evidence');
    const sourcePath = path.resolve(requireString(evidence.sourcePath, 'modelIREvidence.sourcePath'));
    const hash = requireString(evidence.hash, 'modelIREvidence.hash');
    const sizeBytes = requirePositiveInteger(evidence.sizeBytes, 'modelIREvidence.sizeBytes');
    const packPath = toPosix(path.join(
      'artifacts',
      'evidence',
      `model-ir-${hash.slice('sha256:'.length, 'sha256:'.length + 12)}.json`
    ));
    const artifact = {
      artifactId: artifactId('source-truth-evidence', packPath, hash),
      role: 'source-truth-evidence',
      path: packPath,
      hash,
      sizeBytes,
      sourcePath,
    };
    artifacts.push(artifact);
    modelIREvidenceArtifactId = artifact.artifactId;
  }
  const programBundleArtifact = {
    artifactId: artifactId('program-bundle', 'artifacts/program-bundle.json', `sha256:${sha256Hex(input.programBundleRaw)}`),
    role: 'program-bundle',
    path: 'artifacts/program-bundle.json',
    hash: `sha256:${sha256Hex(input.programBundleRaw)}`,
    sizeBytes: new TextEncoder().encode(input.programBundleRaw).byteLength,
    sourcePath: input.programBundlePath,
  };
  artifacts.push(programBundleArtifact);
  const ids = new Set();
  for (const artifact of artifacts) {
    if (ids.has(artifact.artifactId)) throw new Error(`Forge produced duplicate artifactId "${artifact.artifactId}".`);
    ids.add(artifact.artifactId);
  }
  return {
    ...input,
    stage: 'normalize',
    ok: true,
    manifestHash,
    programBundleHash: programBundleArtifact.hash,
    artifacts,
    qualificationEvidence,
    modelIREvidenceArtifactId,
    programBundleArtifactId: programBundleArtifact.artifactId,
  };
}

/** Stage 3: Analyze source facts into hardware-independent ModelIR. */
export function stageAnalyze(normalized) {
  const source = requireObject(normalized, 'normalized source');
  const manifest = source.manifest;
  if (source.modelIR !== null && source.modelIR !== undefined) {
    const validation = validateModelIR(source.modelIR);
    if (!validation.ok) {
      throw new Error(`Forge analyze rejected supplied ModelIR: ${validation.errors.join('; ')}`);
    }
    if (source.modelIR.modelId !== manifest.modelId) {
      throw new Error(
        `Forge ModelIR modelId "${source.modelIR.modelId}" does not match manifest modelId "${manifest.modelId}".`
      );
    }
    let modelIR = source.modelIR;
    const sourceIdentity = modelIR.sourceIdentity;
    if (source.modelIR.schema === 'doppler.model-ir/v2') {
      if (!source.modelIREvidenceArtifactId) {
        throw new Error('Forge ModelIR v2 requires packaged source-truth evidence.');
      }
      const artifactIdentity = requireObject(manifest.artifactIdentity, 'manifest.artifactIdentity');
      const checkpointId = requireString(
        artifactIdentity.sourceCheckpointId,
        'manifest.artifactIdentity.sourceCheckpointId'
      );
      if (sourceIdentity.checkpointId !== checkpointId) {
        throw new Error(
          `Forge ModelIR checkpointId "${sourceIdentity.checkpointId}" does not match manifest source checkpoint "${checkpointId}".`
        );
      }
      if (artifactIdentity.sourceRepo !== undefined
        && sourceIdentity.repository !== artifactIdentity.sourceRepo) {
        throw new Error('Forge ModelIR repository does not match manifest artifact identity.');
      }
      if (artifactIdentity.sourceRevision !== undefined
        && sourceIdentity.revision !== artifactIdentity.sourceRevision) {
        throw new Error('Forge ModelIR revision does not match manifest artifact identity.');
      }
      modelIR = promoteQualifiedModelIRV2(modelIR, source.programBundle);
    }
    return {
      stage: 'analyze',
      ok: true,
      modelIR,
      modelIRHash: hashModelIR(modelIR),
      normalized: source,
    };
  }
  assertModelTopologyRepresentable(manifest);
  const architecture = requireObject(manifest.architecture, 'manifest.architecture');
  const inference = requireObject(manifest.inference, 'manifest.inference');
  const attention = requireObject(inference.attention, 'manifest.inference.attention');
  const normalization = requireObject(inference.normalization, 'manifest.inference.normalization');
  const ffn = requireObject(inference.ffn, 'manifest.inference.ffn');
  const output = requireObject(inference.output, 'manifest.inference.output');
  const layerPattern = requireObject(inference.layerPattern, 'manifest.inference.layerPattern');
  const session = requireObject(inference.session, 'manifest.inference.session');
  const tensors = requireObject(manifest.tensors, 'manifest.tensors');
  const numLayers = requirePositiveInteger(architecture.numLayers, 'manifest.architecture.numLayers');
  const tensorRoles = Object.fromEntries(Object.entries(tensors).map(([name, tensor]) => {
    requireObject(tensor, `manifest.tensors.${name}`);
    return [name, {
      role: requireString(tensor.role, `manifest.tensors.${name}.role`),
      shape: Array.isArray(tensor.shape) ? tensor.shape : null,
      semanticDtype: requireString(tensor.dtype, `manifest.tensors.${name}.dtype`),
    }];
  }));
  const modelIR = createModelIR({
    modelId: requireString(manifest.modelId, 'manifest.modelId'),
    architecture: requireString(manifest.modelType, 'manifest.modelType'),
    vocabSize: requirePositiveInteger(architecture.vocabSize, 'manifest.architecture.vocabSize'),
    hiddenSize: requirePositiveInteger(architecture.hiddenSize, 'manifest.architecture.hiddenSize'),
    numLayers,
    sourceIdentity: {
      manifestArtifactId: source.artifacts.find((artifact) => artifact.role === 'manifest')?.artifactId,
      manifestHash: source.manifestHash,
      sourceCheckpointId: requireString(manifest.artifactIdentity?.sourceCheckpointId, 'manifest.artifactIdentity.sourceCheckpointId'),
    },
    tensorRoles,
    layers: Array.from({ length: numLayers }, (_, index) => ({
      index,
      type: resolveLayerType(index, layerPattern),
      attention: {
        causal: attention.causal,
        slidingWindow: resolveLayerType(index, layerPattern) === 'local-attention'
          ? requirePositiveInteger(attention.slidingWindow, 'manifest.inference.attention.slidingWindow')
          : null,
      },
    })),
    attentionGeometry: {
      numHeads: requirePositiveInteger(architecture.numAttentionHeads, 'manifest.architecture.numAttentionHeads'),
      numKvHeads: requirePositiveInteger(architecture.numKeyValueHeads, 'manifest.architecture.numKeyValueHeads'),
      headDim: requirePositiveInteger(architecture.headDim, 'manifest.architecture.headDim'),
      qkNorm: attention.queryKeyNorm === true,
    },
    normalization: {
      type: normalization.rmsNormWeightOffset === true ? 'gemma-rmsnorm' : 'rmsnorm',
      eps: normalization.rmsNormEps,
    },
    rope: {
      dimension: requirePositiveInteger(architecture.headDim, 'manifest.architecture.headDim'),
      baseFreq: requirePositiveInteger(inference.rope?.ropeTheta, 'manifest.inference.rope.ropeTheta'),
      localBaseFreq: requirePositiveInteger(inference.rope?.ropeLocalTheta, 'manifest.inference.rope.ropeLocalTheta'),
    },
    ffn: {
      type: ffn.gatedActivation === true ? `gated-${requireString(ffn.activation, 'manifest.inference.ffn.activation')}` : requireString(ffn.activation, 'manifest.inference.ffn.activation'),
      intermediateSize: requirePositiveInteger(architecture.intermediateSize, 'manifest.architecture.intermediateSize'),
    },
    outputTopology: {
      headType: 'causal-lm',
      tieWeights: output.tieWordEmbeddings === true,
    },
    phases: ['prefill', 'decode'],
    session,
  });
  return { stage: 'analyze', ok: true, modelIR, modelIRHash: hashModelIR(modelIR), normalized: source };
}

function dtypeByteWidth(dtype, label) {
  const normalized = requireString(dtype, label).toLowerCase();
  if (normalized === 'f16' || normalized === 'float16' || normalized === 'bf16') return 2;
  if (normalized === 'f32' || normalized === 'float32') return 4;
  throw new Error(`Forge cannot size state with unsupported dtype "${dtype}" at ${label}.`);
}

function resolveModelIRSpecialization(modelIR, manifest) {
  if (modelIR.schema !== 'doppler.model-ir/v2') {
    return {
      hiddenSize: modelIR.hiddenSize,
      vocabSize: modelIR.vocabSize,
      kvElementsPerToken: modelIR.numLayers
        * modelIR.attentionGeometry.numKvHeads
        * modelIR.attentionGeometry.headDim
        * 2,
      recurrentStateBytes: 0,
      convolutionalStateBytes: 0,
    };
  }

  const loweredGenerate = modelIR.entryPoints.filter((entryPoint) => (
    entryPoint.kind === 'generate'
      && entryPoint.status === 'lowered'
      && entryPoint.phases.includes('prefill')
      && entryPoint.phases.includes('decode')
  ));
  if (loweredGenerate.length !== 1) {
    throw new Error('Forge requires exactly one lowered ModelIR v2 generate entry point with prefill and decode.');
  }
  const entryPoint = loweredGenerate[0];
  if (!modelIR.supportScope.loweredEntryPoints.includes(entryPoint.id)) {
    throw new Error('Forge generate entry point is absent from ModelIR supportScope.loweredEntryPoints.');
  }
  const component = modelIR.components.find((candidate) => candidate.id === entryPoint.componentId);
  const schedule = modelIR.blockSchedules.find((candidate) => candidate.componentId === entryPoint.componentId);
  if (!component || !schedule) {
    throw new Error('Forge cannot resolve the lowered entry point component and block schedule.');
  }
  const hiddenSize = requirePositiveInteger(component.properties.hiddenSize, `${component.id}.properties.hiddenSize`);
  const vocabSize = requirePositiveInteger(component.properties.vocabSize, `${component.id}.properties.vocabSize`);
  const numLayers = requirePositiveInteger(component.properties.numLayers, `${component.id}.properties.numLayers`);
  if (schedule.blocks.length !== numLayers) {
    throw new Error(`Forge block schedule "${schedule.id}" does not contain ${numLayers} blocks.`);
  }

  const classes = new Map(modelIR.blockClasses.map((blockClass) => [blockClass.id, blockClass]));
  let kvElementsPerToken = 0;
  let recurrentStateElements = 0;
  let convolutionalStateElements = 0;
  for (const block of schedule.blocks) {
    const blockClass = classes.get(block.blockClassId);
    if (!blockClass) throw new Error(`Forge cannot resolve block class "${block.blockClassId}".`);
    if (blockClass.kind === 'full-attention' || blockClass.kind === 'local-attention') {
      kvElementsPerToken += requirePositiveInteger(
        blockClass.geometry.numKvHeads,
        `${blockClass.id}.geometry.numKvHeads`
      ) * requirePositiveInteger(blockClass.geometry.headDim, `${blockClass.id}.geometry.headDim`) * 2;
    }
    if (blockClass.kind === 'linear-recurrent-attention') {
      const valueHeads = requirePositiveInteger(blockClass.geometry.valueHeads, `${blockClass.id}.geometry.valueHeads`);
      const keyHeadDim = requirePositiveInteger(blockClass.geometry.keyHeadDim, `${blockClass.id}.geometry.keyHeadDim`);
      const valueHeadDim = requirePositiveInteger(blockClass.geometry.valueHeadDim, `${blockClass.id}.geometry.valueHeadDim`);
      const keyHeads = requirePositiveInteger(blockClass.geometry.keyHeads, `${blockClass.id}.geometry.keyHeads`);
      const convState = modelIR.stateSpaces.find((state) => state.kind === 'convolutional');
      const kernelSize = requirePositiveInteger(
        convState?.contract?.kernelSize,
        'ModelIR convolutional state contract.kernelSize'
      );
      recurrentStateElements += valueHeads * keyHeadDim * valueHeadDim;
      convolutionalStateElements += (keyHeads * keyHeadDim * 2 + valueHeads * valueHeadDim) * kernelSize;
    }
  }
  const recurrentState = modelIR.stateSpaces.find((state) => state.kind === 'recurrent');
  if (recurrentStateElements > 0 && !recurrentState) {
    throw new Error('Forge heterogeneous lowering requires a recurrent state-space contract.');
  }
  const recurrentStateBytes = recurrentStateElements > 0
    ? recurrentStateElements * dtypeByteWidth(recurrentState.contract.dtype, 'ModelIR recurrent state contract.dtype')
    : 0;
  const convolutionalState = modelIR.stateSpaces.find((state) => state.kind === 'convolutional');
  return {
    hiddenSize,
    vocabSize,
    kvElementsPerToken,
    recurrentStateBytes,
    convolutionalStateBytes: convolutionalStateElements > 0
      ? convolutionalStateElements * dtypeByteWidth(
        convolutionalState.contract.dtype,
        'ModelIR convolutional state contract.dtype'
      )
      : 0,
  };
}

/** Stage 4: Lower the closed Program Bundle into phase commands. */
export function stageLower(analyzed) {
  const validation = validateModelIR(analyzed?.modelIR);
  if (!validation.ok) throw new Error(`Forge lower requires valid ModelIR: ${validation.errors.join('; ')}`);
  const bundle = analyzed.normalized.programBundle;
  const steps = bundle.execution?.steps;
  if (!Array.isArray(steps) || steps.length === 0) throw new Error('Forge lower requires expanded Program Bundle execution steps.');
  const buildPhase = (phase) => [{
    kind: 'program-phase',
    phase,
    executionGraphHash: bundle.execution.graphHash,
    declaredStepIds: steps
      .filter((step) => step.phase === phase || step.phase === 'both')
      .map((step) => step.id),
  }];
  return {
    ...analyzed,
    stage: 'lower',
    ok: true,
    loweredProgram: {
      execution: bundle.execution,
      phases: { prefill: buildPhase('prefill'), decode: buildPhase('decode') },
    },
  };
}

function buildQualificationRecords(lowered) {
  const normalized = lowered.normalized;
  const referenceArtifact = normalized.artifacts.find((artifact) => artifact.role === 'reference-report');
  if (!referenceArtifact) throw new Error('Forge requires a packaged reference-report artifact.');
  const transcript = normalized.programBundle.referenceTranscript;
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

/** Stage 5: Specialize exactly the execution plan present in source evidence. */
export function stageSpecialize(lowered) {
  const modelIR = lowered.modelIR;
  const normalized = lowered.normalized;
  const manifest = normalized.manifest;
  const session = manifest.inference.session;
  const modules = normalized.programBundle.wgslModules;
  if (!Array.isArray(modules) || modules.length === 0) throw new Error('Forge specialize requires a non-empty WGSL closure.');
  const moduleArtifactByHash = new Map(normalized.artifacts
    .filter((artifact) => artifact.role === 'wgsl-source')
    .map((artifact) => [artifact.hash, artifact]));
  const wgslModules = modules.map((module) => {
    const sourceArtifact = moduleArtifactByHash.get(module.sourceHash);
    if (!sourceArtifact) throw new Error(`Forge cannot bind WGSL source bytes for module "${module.id}".`);
    return {
      id: module.id,
      file: module.file,
      entry: module.entry,
      digest: module.digest,
      sourceHash: module.sourceHash,
      sourceArtifactId: sourceArtifact.artifactId,
      metadata: module.metadata,
    };
  });
  const activationDtype = requireString(session.compute?.defaults?.activationDtype, 'manifest.inference.session.compute.defaults.activationDtype');
  const kvDtype = requireString(session.kvcache?.kvDtype, 'manifest.inference.session.kvcache.kvDtype');
  const weightDtype = requireString(manifest.quantizationInfo?.weights, 'manifest.quantizationInfo.weights');
  const specialization = resolveModelIRSpecialization(modelIR, manifest);
  if (modelIR.schema === 'doppler.model-ir/v2' && normalized.initialExecutionIdentity === null) {
    throw new Error('Forge requires a pre-dispatch initial execution identity for ModelIR v2 specialization.');
  }
  const requiresSubgroups = wgslModules.some((module) => module.metadata?.requiresSubgroups === true);
  const bytesPerActivation = activationDtype === 'f16' ? 2 : 4;
  const bytesPerKv = kvDtype === 'f16' ? 2 : 4;
  const bufferSlots = [
    {
      slotId: 'input_tokens', role: 'token-ids', scope: 'transient', owner: 'runtime',
      usage: ['storage', 'copy-dst'],
      size: { op: 'affine', constantBytes: 0, terms: { seqLen: 4 }, alignment: 256, minimumBytes: 256 },
    },
    {
      slotId: 'hidden_state', role: 'activation', scope: 'layer-recycled', owner: 'program',
      usage: ['storage'],
      size: { op: 'affine', constantBytes: 0, terms: { seqLen: specialization.hiddenSize * bytesPerActivation }, alignment: 256, minimumBytes: 256 },
    },
  ];
  if (specialization.kvElementsPerToken > 0) {
    bufferSlots.push({
      slotId: 'kv_cache', role: 'kv', scope: 'session', owner: 'program',
      usage: ['storage'],
      size: { op: 'affine', constantBytes: 0, terms: { maxSeqLen: specialization.kvElementsPerToken * bytesPerKv }, alignment: 256, minimumBytes: 256 },
    });
  }
  if (specialization.recurrentStateBytes > 0) {
    bufferSlots.push({
      slotId: 'recurrent_state', role: 'recurrent-state', scope: 'session', owner: 'program',
      usage: ['storage', 'copy-dst'],
      size: { op: 'constant', bytes: specialization.recurrentStateBytes },
    });
  }
  if (specialization.convolutionalStateBytes > 0) {
    bufferSlots.push({
      slotId: 'convolutional_state', role: 'convolutional-state', scope: 'session', owner: 'program',
      usage: ['storage', 'copy-dst'],
      size: { op: 'constant', bytes: specialization.convolutionalStateBytes },
    });
  }
  bufferSlots.push({
    slotId: 'logits', role: 'logits', scope: 'transient', owner: 'program',
    usage: ['storage', 'copy-src'],
    size: { op: 'constant', bytes: specialization.vocabSize * 4 },
  });

  const targetPlanFields = {
    targetId: `webgpu-${activationDtype}-${kvDtype}-${requiresSubgroups ? 'subgroups' : 'portable'}`,
    modelId: modelIR.modelId,
    modelIRHash: lowered.modelIRHash,
    executionGraphHash: normalized.programBundle.execution.graphHash,
    programBundleHash: normalized.programBundleHash,
    capabilityPredicate: {
      requiresF16: activationDtype === 'f16' || kvDtype === 'f16',
      requiresSubgroups,
      minBufferSize: Math.max(...manifest.shards.map((shard) => requirePositiveInteger(shard.size, 'manifest.shards[].size'))),
    },
    dtypes: { activation: activationDtype, kv: kvDtype, weight: weightDtype },
    fusions: [],
    kernelClosure: wgslModules.map((module) => ({
      moduleId: module.id,
      digest: module.digest,
      sourceHash: module.sourceHash,
    })),
    memoryLayout: {
      kvCacheLayout: requireString(session.kvcache.layout, 'manifest.inference.session.kvcache.layout'),
      bufferSlots,
    },
    phases: lowered.loweredProgram.phases,
    qualification: buildQualificationRecords(lowered),
  };
  let targetPlan;
  if (normalized.initialExecutionIdentity !== null) {
    const identityValidation = validateInitialExecutionIdentity(normalized.initialExecutionIdentity);
    if (!identityValidation.ok) {
      throw new Error(`Forge rejected initial execution identity: ${identityValidation.errors.join('; ')}`);
    }
    targetPlan = createTargetPlanV2({
      ...targetPlanFields,
      initialExecutionIdentity: normalized.initialExecutionIdentity,
    });
  } else {
    targetPlan = createTargetPlan(targetPlanFields);
  }
  return {
    ...lowered, stage: 'specialize', ok: true,
    targetPlans: [targetPlan], targetPlanHashes: [hashTargetPlan(targetPlan)], wgslModules,
  };
}

/** Stage 6: Record the selected prequalified search result without runtime search. */
export function stageSearch(specialized) {
  return {
    ...specialized, stage: 'search', ok: true,
    searchReceipt: {
      candidateTargetPlanHashes: [...specialized.targetPlanHashes],
      selectedTargetPlanHashes: [...specialized.targetPlanHashes],
      policy: 'closed-program-source-plan',
    },
  };
}

function assertTargetPlanMatchesInitialExecutionIdentity(plan) {
  if (plan.schema !== 'doppler.target-plan/v2') return;
  const plannedKernels = plan.kernelClosure
    .map(({ moduleId, digest }) => ({ moduleId, digest }))
    .sort((left, right) => left.moduleId.localeCompare(right.moduleId));
  const observedKernels = plan.initialExecutionIdentity.kernelClosure
    .map(({ moduleId, digest }) => ({ moduleId, digest }))
    .sort((left, right) => left.moduleId.localeCompare(right.moduleId));
  if (hashStable(plannedKernels) !== hashStable(observedKernels)) {
    throw new Error('Forge verify found a TargetPlan kernel closure different from the observed initial execution.');
  }
  for (const lane of ['activation', 'kv']) {
    if (plan.dtypes[lane] !== plan.initialExecutionIdentity.dtypeLane[lane]) {
      throw new Error(`Forge verify found TargetPlan dtype lane "${lane}" different from initial execution.`);
    }
  }
  if (hashStable(plan.fusions) !== hashStable(plan.initialExecutionIdentity.fusionSet)) {
    throw new Error('Forge verify found a TargetPlan fusion set different from the observed initial execution.');
  }
  if (plan.memoryLayout.kvCacheLayout !== plan.initialExecutionIdentity.kvLayout.layout) {
    throw new Error('Forge verify found a TargetPlan KV layout different from the observed initial execution.');
  }
}

/** Stage 7: Verify graph, ModelIR, plan, and kernel closure bindings. */
export function stageVerify(searched) {
  const modelValidation = validateModelIR(searched.modelIR);
  if (!modelValidation.ok) throw new Error(`Forge verify rejected ModelIR: ${modelValidation.errors.join('; ')}`);
  const moduleIds = new Set(searched.wgslModules.map((module) => module.id));
  for (const plan of searched.targetPlans) {
    const validation = validateTargetPlan(plan);
    if (!validation.ok) throw new Error(`Forge verify rejected TargetPlan: ${validation.errors.join('; ')}`);
    if (plan.modelIRHash !== searched.modelIRHash) throw new Error('Forge verify found a TargetPlan bound to a different ModelIR.');
    if (plan.kernelClosure.some((kernel) => !moduleIds.has(kernel.moduleId))) {
      throw new Error('Forge verify found a TargetPlan kernel outside the WGSL closure.');
    }
    assertTargetPlanMatchesInitialExecutionIdentity(plan);
  }
  return { ...searched, stage: 'verify', ok: true, verificationReceipt: { modelIRHash: searched.modelIRHash, targetPlanHashes: searched.targetPlanHashes } };
}

/** Stage 8: Require passed, packaged execution evidence for every plan. */
export function stageQualify(verified) {
  for (const plan of verified.targetPlans) {
    if (!plan.qualification.every((record) => record.status === 'passed')) {
      throw new Error(`Forge qualify rejected target "${plan.targetId}".`);
    }
  }
  return { ...verified, stage: 'qualify', ok: true, qualificationReceipt: { targetIds: verified.targetPlans.map((plan) => plan.targetId) } };
}

/** Stage 9: Package the deterministic unsigned envelope. */
export function stagePackage(qualified) {
  const normalized = qualified.normalized;
  const artifacts = normalized.artifacts.map(stripForgeOnlyArtifactFields);
  const findIds = (role) => artifacts.filter((artifact) => artifact.role === role).map((artifact) => artifact.artifactId);
  const pack = buildPackV2({
    modelId: qualified.modelIR.modelId,
    createdAtUtc: requireString(normalized.programBundle.createdAtUtc, 'Program Bundle createdAtUtc'),
    modelIR: qualified.modelIR,
    targetPlans: qualified.targetPlans,
    wgslModules: qualified.wgslModules,
    artifacts,
    program: {
      schema: PACK_V2_PROGRAM_SCHEMA_ID,
      programBundleHash: normalized.programBundleHash,
      programBundleArtifactId: normalized.programBundleArtifactId,
      executionGraphHash: normalized.programBundle.execution.graphHash,
      manifestArtifactId: findIds('manifest')[0],
      ...(normalized.modelIREvidenceArtifactId
        ? { modelIREvidenceArtifactId: normalized.modelIREvidenceArtifactId }
        : {}),
      tokenizerArtifactIds: findIds('tokenizer'),
      weightArtifactIds: findIds('weight-shard'),
      execution: normalized.programBundle.execution,
      referenceTranscript: normalized.programBundle.referenceTranscript,
    },
  });
  return { ...qualified, stage: 'package', ok: true, pack };
}

/** Stage 10: Ed25519-sign the immutable semantic root. */
export async function stageSign(packaged, signer) {
  const pack = await signPackV2(packaged.pack, signer);
  return { ...packaged, stage: 'sign', ok: true, pack, semanticRoot: pack.semanticRoot };
}

/** Runs the complete Forge pipeline in its fixed constitutional order. */
export async function runForgePipeline(input, signer) {
  const inspected = await stageInspect(input);
  const normalized = stageNormalize(inspected.data);
  const analyzed = stageAnalyze(normalized);
  const lowered = stageLower(analyzed);
  const specialized = stageSpecialize(lowered);
  const searched = stageSearch(specialized);
  const verified = stageVerify(searched);
  const qualified = stageQualify(verified);
  const packaged = stagePackage(qualified);
  const signed = await stageSign(packaged, signer);
  return {
    pack: signed.pack,
    stages: [inspected, normalized, analyzed, lowered, specialized, searched, verified, qualified, packaged, signed]
      .map((stage) => ({ stage: stage.stage, ok: stage.ok })),
  };
}
