import { DOPPLER_VERSION } from '../../version.js';
import { computeCanonicalSha256 } from '../../formats/canonical-hash.js';
import { isNodeRuntime } from '../../storage/runtime-env.js';
import { assertExecutionAllowed } from '../runtime/resolution-policy.js';

const GENERATION_EVIDENCE_SCHEMA = 'doppler_generation_evidence/v1';
const GENERATION_TRANSCRIPT_SCHEMA = 'doppler_generation_transcript/v1';
const EMBEDDING_EVIDENCE_SCHEMA = 'doppler_embedding_evidence/v1';
export const RERANK_EVIDENCE_SCHEMA = 'doppler_rerank_evidence/v1';
const RUNTIME_PROFILE_SCHEMA = 'doppler_runtime_profile/v1';
const RESOLUTION_IDENTITY_SCHEMA = 'doppler.resolution-identity/v1';
const EXECUTION_IDENTITY_SCHEMA = 'doppler.resolved-execution-identity/v1';

export function hashEvidenceValue(value) {
  return computeCanonicalSha256(value);
}

function cleanEvidenceString(value) {
  const text = String(value ?? '').trim();
  return text || null;
}

export function normalizeSha256Identity(value, label) {
  const normalized = cleanEvidenceString(value)?.toLowerCase() ?? '';
  const digest = normalized.startsWith('sha256:') ? normalized : `sha256:${normalized}`;
  if (!/^sha256:[0-9a-f]{64}$/.test(digest)) {
    throw new Error(`Doppler generation evidence requires ${label} as a SHA-256 digest.`);
  }
  return digest;
}

function buildAdapterIdentity(deviceInfo = {}) {
  return {
    vendor: cleanEvidenceString(deviceInfo.vendor),
    architecture: cleanEvidenceString(deviceInfo.architecture),
    device: cleanEvidenceString(deviceInfo.device),
    description: cleanEvidenceString(deviceInfo.description),
  };
}

export function buildGenerationBackendIdentity({
  deviceInfo = null,
  kernelCapabilities = null,
  stats = null,
} = {}) {
  const capabilities = kernelCapabilities && typeof kernelCapabilities === 'object'
    ? kernelCapabilities
    : {};
  const adapter = deviceInfo && typeof deviceInfo === 'object'
    ? deviceInfo
    : (capabilities.adapterInfo || {});
  const executionPlan = stats?.executionPlan || null;
  const finalPlanId = executionPlan?.finalActivePlanId
    || executionPlan?.activePlanIdAtStart
    || null;
  const primaryPlan = executionPlan?.primary || null;
  const fallbackPlan = executionPlan?.fallback || null;
  const finalPlan = finalPlanId && fallbackPlan?.id === finalPlanId
    ? fallbackPlan
    : primaryPlan;
  return {
    backend: 'webgpu',
    adapter: buildAdapterIdentity(adapter),
    hasF16: capabilities.hasF16 === true,
    hasSubgroups: capabilities.hasSubgroups === true,
    maxBufferSize: Number(capabilities.maxBufferSize || 0),
    deviceEpoch: Number(capabilities.deviceEpoch || 0),
    kernelPathId: cleanEvidenceString(stats?.kernelPathId || finalPlan?.kernelPathId),
    kernelPathSource: cleanEvidenceString(stats?.kernelPathSource || finalPlan?.kernelPathSource),
    executionPlanId: cleanEvidenceString(finalPlanId),
    activationDtype: cleanEvidenceString(finalPlan?.activationDtype),
  };
}

export async function buildResolutionIdentity({
  logicalModelId,
  modelId,
  manifestHash,
  resolvedRuntimeSessionId,
  activeAdapter,
  backendIdentity,
  resolutionPolicy,
}) {
  const resolvedModelId = cleanEvidenceString(modelId);
  const logicalId = cleanEvidenceString(logicalModelId);
  if (!resolvedModelId || !logicalId) {
    throw new Error('Doppler runtime evidence requires logical and resolved model IDs.');
  }
  const resolvedArtifactVariantId = normalizeSha256Identity(manifestHash, 'manifestHash');
  const runtimeSessionId = normalizeSha256Identity(
    resolvedRuntimeSessionId,
    'resolvedRuntimeSessionId'
  );
  const runtimeIdentity = {
    package: 'doppler-gpu',
    version: DOPPLER_VERSION,
    surface: isNodeRuntime() ? 'node' : 'browser',
  };
  const executionIdentity = {
    schema: EXECUTION_IDENTITY_SCHEMA,
    runtime: runtimeIdentity,
    resolvedRuntimeSessionId: runtimeSessionId,
    activeAdapter: cleanEvidenceString(activeAdapter?.name),
    activeAdapterId: cleanEvidenceString(activeAdapter?.id),
    activeAdapterDigest: activeAdapter?.digest ?? null,
    backendIdentity,
  };
  const resolvedExecutionId = await hashEvidenceValue(executionIdentity);
  assertExecutionAllowed(resolutionPolicy, resolvedExecutionId);
  return {
    resolvedModelId,
    resolvedArtifactVariantId,
    runtimeSessionId,
    runtimeIdentity,
    executionIdentity,
    resolution: {
      schema: RESOLUTION_IDENTITY_SCHEMA,
      logicalModelId: logicalId,
      resolvedArtifactVariantId,
      resolvedExecutionId,
    },
  };
}

export async function buildGenerationEvidence({
  outputText,
  tokenIds,
  generationConfig,
  logicalModelId,
  modelId,
  manifestHash,
  resolvedRuntimeSessionId,
  activeAdapter,
  backendIdentity,
  stats,
  resolutionPolicy,
} = {}) {
  if (typeof outputText !== 'string') {
    throw new Error('Doppler generation evidence requires outputText.');
  }
  if (!Array.isArray(tokenIds) || tokenIds.some((tokenId) => !Number.isInteger(tokenId) || tokenId < 0)) {
    throw new Error('Doppler generation evidence requires non-negative integer tokenIds.');
  }
  const transcript = {
    schema: GENERATION_TRANSCRIPT_SCHEMA,
    outputText,
    tokenIds: [...tokenIds],
  };
  const generationConfigHash = await hashEvidenceValue(generationConfig);
  const transcriptHash = await hashEvidenceValue(transcript);
  const backendIdentityHash = await hashEvidenceValue(backendIdentity);
  const identity = await buildResolutionIdentity({
    logicalModelId,
    modelId,
    manifestHash,
    resolvedRuntimeSessionId,
    activeAdapter,
    backendIdentity,
    resolutionPolicy,
  });
  const runtimeProfile = {
    schema: RUNTIME_PROFILE_SCHEMA,
    runtime: identity.runtimeIdentity,
    model: {
      modelId: identity.resolvedModelId,
      manifestHash: identity.resolvedArtifactVariantId,
      activeAdapter: cleanEvidenceString(activeAdapter?.name),
      activeAdapterId: cleanEvidenceString(activeAdapter?.id),
      activeAdapterDigest: activeAdapter?.digest ?? null,
    },
    resolvedRuntimeSessionId: identity.runtimeSessionId,
    backendIdentity,
  };
  const runtimeProfileHash = await hashEvidenceValue(runtimeProfile);
  return {
    schema: GENERATION_EVIDENCE_SCHEMA,
    outputText,
    tokenIds: [...tokenIds],
    transcript,
    transcriptHash,
    generationConfig,
    generationConfigHash,
    resolution: identity.resolution,
    executionIdentity: identity.executionIdentity,
    runtimeProfile,
    runtimeProfileHash,
    backendIdentity,
    backendIdentityHash,
    stats: stats && typeof stats === 'object' ? stats : null,
  };
}

export async function buildEmbeddingEvidence({
  prompt,
  result,
  logicalModelId,
  modelId,
  manifestHash,
  resolvedRuntimeSessionId,
  activeAdapter,
  backendIdentity,
  stats,
  resolutionPolicy,
}) {
  const embedding = Array.from(result?.embedding || [], Number);
  if (embedding.length === 0 || embedding.some((value) => !Number.isFinite(value))) {
    throw new Error('Doppler embedding evidence requires a finite non-empty embedding.');
  }
  const tokens = Array.from(result?.tokens || [], Number);
  if (tokens.some((tokenId) => !Number.isInteger(tokenId) || tokenId < 0)) {
    throw new Error('Doppler embedding evidence requires non-negative integer tokens.');
  }
  const seqLen = Number(result?.seqLen ?? tokens.length);
  if (!Number.isInteger(seqLen) || seqLen < 0) {
    throw new Error('Doppler embedding evidence requires a non-negative integer seqLen.');
  }
  const embeddingMode = cleanEvidenceString(result?.embeddingMode);
  if (!embeddingMode) {
    throw new Error('Doppler embedding evidence requires embeddingMode.');
  }
  const outputIdentity = {
    embedding,
    tokens,
    seqLen,
    embeddingMode,
  };
  const identity = await buildResolutionIdentity({
    logicalModelId,
    modelId,
    manifestHash,
    resolvedRuntimeSessionId,
    activeAdapter,
    backendIdentity,
    resolutionPolicy,
  });
  return {
    schema: EMBEDDING_EVIDENCE_SCHEMA,
    ...result,
    inputHash: await hashEvidenceValue({ text: String(prompt) }),
    outputHash: await hashEvidenceValue(outputIdentity),
    resolution: identity.resolution,
    executionIdentity: identity.executionIdentity,
    backendIdentity,
    backendIdentityHash: await hashEvidenceValue(backendIdentity),
    stats: stats && typeof stats === 'object' ? stats : null,
  };
}
