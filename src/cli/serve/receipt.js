import crypto from 'node:crypto';
import { DOPPLER_VERSION } from '../../index.js';

export function generateCompletionId() {
  return `chatcmpl-${crypto.randomBytes(12).toString('base64url')}`;
}

function sha256Hex(value) {
  return crypto.createHash('sha256').update(value).digest('hex');
}

function stableJson(value) {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) {
    return `[${value.map((entry) => stableJson(entry)).join(',')}]`;
  }
  const entries = Object.keys(value)
    .sort()
    .filter((key) => value[key] !== undefined)
    .map((key) => `${JSON.stringify(key)}:${stableJson(value[key])}`);
  return `{${entries.join(',')}}`;
}

function buildSha256Evidence(value) {
  const text = String(value ?? '');
  return {
    algorithm: 'sha256',
    value: sha256Hex(text),
    bytes: Buffer.byteLength(text, 'utf8'),
  };
}

function buildJsonSha256Evidence(value) {
  return buildSha256Evidence(stableJson(value));
}

function normalizeResolutionIdentity(resolution) {
  if (resolution == null) return null;
  const normalized = {
    schema: String(resolution.schema || ''),
    logicalModelId: String(resolution.logicalModelId || '').trim(),
    resolvedArtifactVariantId: String(resolution.resolvedArtifactVariantId || '').toLowerCase(),
    resolvedExecutionId: String(resolution.resolvedExecutionId || '').toLowerCase(),
  };
  if (normalized.schema !== 'doppler.resolution-identity/v1') {
    throw new Error('Doppler serve receipt requires doppler.resolution-identity/v1.');
  }
  if (!normalized.logicalModelId) {
    throw new Error('Doppler serve receipt requires resolution.logicalModelId.');
  }
  for (const field of ['resolvedArtifactVariantId', 'resolvedExecutionId']) {
    if (!/^sha256:[0-9a-f]{64}$/.test(normalized[field])) {
      throw new Error(`Doppler serve receipt requires resolution.${field} as a SHA-256 digest.`);
    }
  }
  return normalized;
}

function buildGenerationOptionsReceipt(generationOptions) {
  return {
    maxTokens: generationOptions.maxTokens ?? null,
    temperature: generationOptions.temperature ?? null,
    topP: generationOptions.topP ?? null,
    topK: generationOptions.topK ?? null,
  };
}

function normalizeRuntimeModelSource(runtimeModel, registryEntry) {
  if (typeof runtimeModel === 'string') {
    return { kind: 'quickstart-registry', modelId: runtimeModel };
  }
  if (runtimeModel && typeof runtimeModel === 'object' && typeof runtimeModel.url === 'string') {
    return { kind: 'url', url: runtimeModel.url };
  }
  if (runtimeModel && typeof runtimeModel === 'object' && runtimeModel.manifest && typeof runtimeModel.manifest === 'object') {
    return {
      kind: 'inline-manifest',
      modelId: typeof runtimeModel.manifest.modelId === 'string'
        ? runtimeModel.manifest.modelId
        : registryEntry.modelId,
      baseUrl: typeof runtimeModel.baseUrl === 'string' ? runtimeModel.baseUrl : null,
    };
  }
  if (runtimeModel && typeof runtimeModel === 'object' && typeof runtimeModel.registryId === 'string') {
    return { kind: 'quickstart-registry', modelId: runtimeModel.registryId };
  }
  return { kind: 'quickstart-registry', modelId: registryEntry.modelId };
}

export function bindLogicalRuntimeModel(runtimeModel, requestedModel) {
  if (typeof runtimeModel === 'string') {
    return { registryId: runtimeModel, logicalModelId: requestedModel };
  }
  if (runtimeModel && typeof runtimeModel === 'object') {
    return {
      ...runtimeModel,
      logicalModelId: runtimeModel.logicalModelId || requestedModel,
    };
  }
  return runtimeModel;
}

function buildServeReceiptBase({
  requestedModel,
  registryEntry,
  messages,
  generationOptions,
  runtimeModel = null,
  resolution = null,
  loadedArtifact = null,
  resolutionUnavailableReason = null,
}) {
  const generation = buildGenerationOptionsReceipt(generationOptions);
  const resolvedIdentity = normalizeResolutionIdentity(resolution);
  const source = normalizeRuntimeModelSource(runtimeModel ?? registryEntry.modelId, registryEntry);
  if (loadedArtifact && (!resolvedIdentity || loadedArtifact.artifactVariantId !== resolvedIdentity.resolvedArtifactVariantId)) {
    throw new Error('Loaded manifest identity differs from generation evidence.');
  }
  const registrySource = source.kind === 'quickstart-registry' && (!loadedArtifact || (
    loadedArtifact.modelId === registryEntry.modelId
    && loadedArtifact.weightCapsuleId === registryEntry.weightCapsuleId
    && loadedArtifact.manifestVariantId === registryEntry.manifestVariantId
  ));
  const modelId = loadedArtifact?.modelId ?? (registrySource ? registryEntry.modelId : null);
  return {
    receiptVersion: 'doppler-serve-receipt-v2',
    schemaVersion: 2,
    surface: 'serve',
    endpoint: '/v1/chat/completions',
    runtime: 'doppler-gpu',
    runtimeVersion: DOPPLER_VERSION,
    runtimePath: 'doppler-gpu.chatText',
    runtimeModelSource: source,
    catalogModelId: registryEntry.modelId,
    modelId,
    requestedModel,
    resolvedModel: modelId,
    resolutionStatus: resolvedIdentity ? 'resolved' : 'unavailable',
    resolution: resolvedIdentity,
    resolutionUnavailableReason: resolvedIdentity
      ? null
      : String(resolutionUnavailableReason || 'not-provided'),
    artifact: {
      format: 'rdrr',
      source: registrySource ? 'quickstart-registry' : (loadedArtifact ? 'loaded-manifest' : source.kind),
      sourceCheckpointId: loadedArtifact?.sourceCheckpointId ?? (registrySource ? registryEntry.sourceCheckpointId : null),
      weightCapsuleId: loadedArtifact?.weightCapsuleId ?? (registrySource ? registryEntry.weightCapsuleId : null),
      manifestVariantId: loadedArtifact?.manifestVariantId ?? (registrySource ? registryEntry.manifestVariantId : null),
      artifactCompleteness: registrySource ? registryEntry.artifactCompleteness : null,
      runtimePromotionState: registrySource ? registryEntry.runtimePromotionState : null,
      weightsRefAllowed: registrySource ? registryEntry.weightsRefAllowed : null,
      hf: registrySource ? registryEntry.hf : null,
    },
    request: {
      messages: { count: messages.length, digest: buildJsonSha256Evidence(messages) },
      generationDigest: buildJsonSha256Evidence(generation),
    },
    generation,
  };
}

export function buildServeReceipt({
  requestedModel,
  registryEntry,
  messages,
  generationOptions,
  outputContent,
  usage,
  runtimeModel = null,
  generationEvidence = null,
}) {
  const baseReceipt = buildServeReceiptBase({
    requestedModel,
    registryEntry,
    messages,
    generationOptions,
    runtimeModel,
    resolution: generationEvidence?.resolution ?? null,
    loadedArtifact: generationEvidence?.loadedArtifact ?? null,
    resolutionUnavailableReason: 'runtime-result-did-not-expose-generation-evidence',
  });
  const outputText = String(outputContent ?? '');
  const rawStats = generationEvidence?.stats;
  const timing = rawStats && typeof rawStats === 'object' ? {
    prefillTimeMs: Number.isFinite(rawStats.prefillTimeMs) ? rawStats.prefillTimeMs : null,
    decodeTimeMs: Number.isFinite(rawStats.decodeTimeMs) ? rawStats.decodeTimeMs : null,
    ttftMs: Number.isFinite(rawStats.ttftMs) ? rawStats.ttftMs : null,
    reusedPrefixTokens: Number.isSafeInteger(rawStats.reusedPrefixTokens) ? rawStats.reusedPrefixTokens : null,
    prefillComputedTokens: Number.isSafeInteger(rawStats.prefillComputedTokens) ? rawStats.prefillComputedTokens : null,
  } : null;
  const receipt = {
    ...baseReceipt,
    status: 'pass',
    output: {
      role: 'assistant',
      digest: buildSha256Evidence(outputText),
      textLength: outputText.length,
      empty: outputText.length === 0,
      ...(Array.isArray(generationEvidence?.tokenIds) ? { tokenIds: [...generationEvidence.tokenIds] } : {}),
    },
    transcript: {
      digest: buildJsonSha256Evidence({
        messages,
        generation: baseReceipt.generation,
        output: outputText,
        usage,
      }),
    },
    usage,
  };
  if (timing) receipt.timing = timing;
  if (generationEvidence?.backendIdentity) receipt.backendIdentity = generationEvidence.backendIdentity;
  return receipt;
}

function normalizeWeightLoadFailure(error) {
  const failure = error?.details?.weightLoadFailure ?? error?.cause?.details?.weightLoadFailure;
  if (!failure || typeof failure !== 'object') return null;
  const limits = failure.deviceLimitFailure;
  const deviceLimitFailure = limits && typeof limits === 'object'
    ? {
      kind: typeof limits.kind === 'string' ? limits.kind : null,
      maxGpuResidentBytes: Number.isFinite(limits.maxGpuResidentBytes) ? limits.maxGpuResidentBytes : null,
      maxStorageBufferBindingSize: Number.isFinite(limits.maxStorageBufferBindingSize)
        ? limits.maxStorageBufferBindingSize
        : null,
      maxBufferSize: Number.isFinite(limits.maxBufferSize) ? limits.maxBufferSize : null,
      maxStorageBuffersPerShaderStage: Number.isFinite(limits.maxStorageBuffersPerShaderStage)
        ? limits.maxStorageBuffersPerShaderStage
        : null,
      largeWeightMaxBytes: Number.isFinite(limits.largeWeightMaxBytes) ? limits.largeWeightMaxBytes : null,
      embeddingKernel: limits.embeddingKernel && typeof limits.embeddingKernel === 'object'
        ? {
          kernel: typeof limits.embeddingKernel.kernel === 'string' ? limits.embeddingKernel.kernel : null,
          entry: typeof limits.embeddingKernel.entry === 'string' ? limits.embeddingKernel.entry : null,
        }
        : null,
      splitKernelExpected: typeof limits.splitKernelExpected === 'boolean' ? limits.splitKernelExpected : null,
      activeSplitKernelMaxSections: Number.isFinite(limits.activeSplitKernelMaxSections)
        ? limits.activeSplitKernelMaxSections
        : null,
      maxSplitEmbeddingSections: Number.isFinite(limits.maxSplitEmbeddingSections)
        ? limits.maxSplitEmbeddingSections
        : null,
      requiredSplitSections: Number.isFinite(limits.requiredSplitSections) ? limits.requiredSplitSections : null,
    }
    : null;
  return {
    tensorName: typeof failure.tensorName === 'string' ? failure.tensorName : null,
    tensorRole: typeof failure.tensorRole === 'string' ? failure.tensorRole : null,
    tensorDtype: typeof failure.tensorDtype === 'string' ? failure.tensorDtype : null,
    tensorShape: Array.isArray(failure.tensorShape) ? [...failure.tensorShape] : null,
    tensorSizeBytes: Number.isFinite(failure.tensorSizeBytes) ? failure.tensorSizeBytes : null,
    tensorLoadStage: typeof failure.tensorLoadStage === 'string' ? failure.tensorLoadStage : null,
    toGPU: typeof failure.toGPU === 'boolean' ? failure.toGPU : null,
    streamedUpload: typeof failure.streamedUpload === 'boolean' ? failure.streamedUpload : null,
    deviceLimitFailure,
  };
}

function normalizeServeFailure(error, registryEntry) {
  const pipelineLoadPhase = typeof error?.details?.pipelineLoadPhase === 'string'
    ? error.details.pipelineLoadPhase
    : null;
  return {
    code: pipelineLoadPhase ? 'pipeline-load-failed' : 'runtime-error',
    stage: pipelineLoadPhase ?? 'runtime',
    message: error?.message || String(error),
    modelId: typeof error?.details?.modelId === 'string' ? error.details.modelId : registryEntry.modelId,
    weightLoadFailure: normalizeWeightLoadFailure(error),
  };
}

export function buildServeFailureReceipt({
  requestedModel,
  registryEntry,
  messages,
  generationOptions,
  error,
  runtimeModel = null,
}) {
  return {
    ...buildServeReceiptBase({
      requestedModel,
      registryEntry,
      messages,
      generationOptions,
      runtimeModel,
      resolutionUnavailableReason: 'execution-failed-before-resolution',
    }),
    status: 'diagnostic',
    failure: normalizeServeFailure(error, registryEntry),
  };
}
