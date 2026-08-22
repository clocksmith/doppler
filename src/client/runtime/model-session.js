import { getKernelCapabilities } from '../../gpu/device.js';
import { formatChatMessages } from '../../inference/pipelines/text/chat-format.js';
import { applyChatTemplate } from '../../inference/pipelines/text/init-chat-templates.js';
import { resolveSamplingConfig } from '../../inference/pipelines/text/sampling-config.js';
import { DOPPLER_VERSION } from '../../version.js';
import { computeCanonicalSha256 } from '../../utils/canonical-hash.js';
import { isNodeRuntime } from '../../utils/runtime-env.js';
import {
  MODEL_INSPECTION_RECEIPT_SCHEMA,
  aggregateWordPerplexity,
  buildComparisonFingerprint,
  buildInspectionTokenRecords,
  listObservationPolicies,
  resolveObservationPolicy,
} from '../inspection.js';
import {
  activateLoRAFromTrainingOutputForPipeline,
  getActiveLoRAForPipeline,
  getActiveLoRAIdentityForPipeline,
  loadLoRAAdapterForPipeline,
  unloadLoRAAdapterForPipeline,
} from './lora.js';
import {
  assertArtifactVariantAllowed,
  assertExecutionAllowed,
  assertExecutionMayStart,
  assertUnreceiptedExecutionAllowed,
  resolveResolutionPolicy,
} from './resolution-policy.js';

export function assertSupportedGenerationOptions(options = {}) {
  if (Array.isArray(options?.stopTokens) && options.stopTokens.length > 0) {
    throw new Error(
      'Doppler generate options do not support stopTokens on this surface. ' +
      'Use stopSequences instead.'
    );
  }
}

function countTokens(pipeline, text) {
  if (!text || typeof text !== 'string') return 0;
  try {
    return pipeline?.tokenizer?.encode(text)?.length ?? 0;
  } catch {
    return 0;
  }
}

function tokenizeText(pipeline, text) {
  if (typeof text !== 'string') {
    throw new Error('Doppler advanced.tokenizeText requires a string.');
  }
  if (!pipeline?.tokenizer || typeof pipeline.tokenizer.encode !== 'function') {
    throw new Error('Loaded Doppler pipeline does not expose tokenizer.encode().');
  }
  const tokenIds = pipeline.tokenizer.encode(text);
  if (!Array.isArray(tokenIds) && !ArrayBuffer.isView(tokenIds)) {
    throw new Error('Loaded Doppler tokenizer.encode() must return token IDs.');
  }
  return Array.from(tokenIds);
}

function tokenizePrompt(pipeline, prompt, options = {}) {
  if (!pipeline?.tokenizer || typeof pipeline.tokenizer.encode !== 'function') {
    throw new Error('Loaded Doppler pipeline does not expose tokenizer.encode().');
  }
  const templateEnabled = options.useChatTemplate === true
    && pipeline?.manifest?.inference?.chatTemplate?.enabled !== false;
  const templateType = templateEnabled
    ? (pipeline?.manifest?.inference?.chatTemplate?.type ?? null)
    : null;
  let text;
  if (typeof prompt === 'string') {
    text = templateType
      ? applyChatTemplate(prompt, templateType, pipeline.modelConfig?.chatTemplateThinking === true ? { thinking: true } : undefined)
      : prompt;
  } else {
    const messages = Array.isArray(prompt) ? prompt : prompt?.messages;
    if (!Array.isArray(messages)) throw new Error('Doppler advanced.tokenizePrompt requires text or chat messages.');
    text = formatChatMessages(messages, templateType, pipeline.modelConfig?.chatTemplateThinking === true ? { thinking: true } : undefined);
  }
  return Array.from(pipeline.tokenizer.encode(text));
}

function resolveChatPromptForUsage(pipeline, messages) {
  const templateType = pipeline?.manifest?.inference?.chatTemplate?.enabled === false
    ? null
    : (pipeline?.manifest?.inference?.chatTemplate?.type ?? null);
  try {
    return formatChatMessages(messages, templateType);
  } catch {
    return messages.map((message) => String(message?.content ?? '')).join('\n');
  }
}

async function collectText(iterable) {
  let output = '';
  for await (const token of iterable) {
    output += token;
  }
  return output;
}

const GENERATION_EVIDENCE_SCHEMA = 'doppler_generation_evidence/v1';
const GENERATION_TRANSCRIPT_SCHEMA = 'doppler_generation_transcript/v1';
const EMBEDDING_EVIDENCE_SCHEMA = 'doppler_embedding_evidence/v1';
const RERANK_EVIDENCE_SCHEMA = 'doppler_rerank_evidence/v1';
const RUNTIME_PROFILE_SCHEMA = 'doppler_runtime_profile/v1';
const RESOLUTION_IDENTITY_SCHEMA = 'doppler.resolution-identity/v1';
const EXECUTION_IDENTITY_SCHEMA = 'doppler.resolved-execution-identity/v1';

function hashEvidenceValue(value) {
  return computeCanonicalSha256(value);
}

function cleanEvidenceString(value) {
  const text = String(value ?? '').trim();
  return text || null;
}

function normalizeSha256Identity(value, label) {
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

function buildGenerationBackendIdentity({
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

async function buildResolutionIdentity({
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

async function buildGenerationEvidence({
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

async function buildEmbeddingEvidence({
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

function resolveUseChatTemplate(pipeline, options) {
  if (typeof options.useChatTemplate === 'boolean') {
    return options.useChatTemplate;
  }
  const runtimeValue = pipeline?.runtimeConfig?.inference?.chatTemplate?.enabled;
  if (typeof runtimeValue === 'boolean') {
    return runtimeValue;
  }
  const modelValue = pipeline?.modelConfig?.chatTemplateEnabled;
  return typeof modelValue === 'boolean' ? modelValue : false;
}

function resolveGenerationConfigEvidence(pipeline, options) {
  const runtimeConfig = pipeline?.runtimeConfig;
  const generation = runtimeConfig?.inference?.generation;
  if (!generation || typeof generation !== 'object' || Array.isArray(generation)) {
    throw new Error('Loaded Doppler pipeline does not expose resolved generation config.');
  }
  const maxTokens = options.maxTokens ?? generation.maxTokens;
  if (!Number.isInteger(maxTokens) || maxTokens <= 0) {
    throw new Error('Resolved Doppler generation maxTokens must be a positive integer.');
  }
  const sampling = resolveSamplingConfig(options, runtimeConfig);
  const stopSequences = options.stopSequences ?? [];
  if (!Array.isArray(stopSequences) || stopSequences.some((value) => typeof value !== 'string')) {
    throw new Error('Resolved Doppler stopSequences must be an array of strings.');
  }
  const useSpeculative = options.useSpeculative ?? generation.useSpeculative ?? null;
  if (useSpeculative !== null && typeof useSpeculative !== 'boolean') {
    throw new Error('Resolved Doppler useSpeculative must be a boolean or null.');
  }
  const seed = options.seed ?? null;
  if (seed !== null && (!Number.isFinite(seed) || seed < 0)) {
    throw new Error('Resolved Doppler seed must be null or a non-negative number.');
  }
  return {
    maxTokens,
    temperature: sampling.temperature,
    topP: sampling.topP,
    topK: sampling.topK,
    repetitionPenalty: sampling.repetitionPenalty,
    repetitionPenaltyWindow: sampling.repetitionPenaltyWindow,
    greedyThreshold: sampling.greedyThreshold,
    suppressSpecialTokens: sampling.suppressSpecialTokens,
    suppressSpecialLikeTokens: sampling.suppressSpecialLikeTokens,
    suppressTokenIds: [...sampling.suppressTokenIds],
    stopSequences: [...stopSequences],
    useChatTemplate: resolveUseChatTemplate(pipeline, options),
    useSpeculative,
    seed,
  };
}

function decodeGeneratedTokens(pipeline, tokenIds) {
  if (!pipeline?.tokenizer || typeof pipeline.tokenizer.decode !== 'function') {
    throw new Error('Loaded Doppler pipeline does not expose tokenizer.decode().');
  }
  return String(pipeline.tokenizer.decode(tokenIds, true, false));
}

function resolveInspectionBrowserIdentity() {
  const navigatorValue = globalThis.navigator;
  return {
    userAgent: navigatorValue?.userAgent ?? '',
    platform: navigatorValue?.platform ?? '',
    language: navigatorValue?.language ?? '',
  };
}

function resolveTokenizerContract(pipeline) {
  const contract = pipeline?.manifest?.tokenizer;
  if (!contract || typeof contract !== 'object' || Array.isArray(contract)) {
    throw new Error('Loaded Doppler manifest does not expose tokenizer identity.');
  }
  return contract;
}

function resolveInspectionGenerationOptions(options, policy) {
  const generation = options?.generation ?? {};
  if (!generation || typeof generation !== 'object' || Array.isArray(generation)) {
    throw new Error('Doppler inspection generation options must be an object.');
  }
  for (const field of ['onToken', 'onLogits', 'profile', 'disableCommandBatching']) {
    if (Object.prototype.hasOwnProperty.call(generation, field)) {
      throw new Error(`Doppler inspection owns generation.${field} through its observation policy.`);
    }
  }
  const resolved = { ...generation };
  if (policy.modifiesExecution) {
    resolved.disableCommandBatching = true;
  }
  if (policy.gpuTimestampQueries) {
    resolved.profile = true;
  }
  return resolved;
}

function assertActiveAdapterUnchanged(pipeline, expected) {
  const observed = getActiveLoRAIdentityForPipeline(pipeline);
  if ((observed?.digest ?? null) !== (expected?.digest ?? null)) {
    throw new Error('Active LoRA adapter changed during Doppler execution.');
  }
}

export function createModelHandle(pipeline, resolved) {
  const resolutionPolicy = resolveResolutionPolicy(resolved.resolutionPolicy);
  assertArtifactVariantAllowed(resolutionPolicy, resolved.manifestHash);
  const assertRaw = (apiName) => assertUnreceiptedExecutionAllowed(
    resolutionPolicy,
    `Doppler model.${apiName}()`
  );

  async function generateWithEvidence(prompt, options = {}) {
    assertExecutionMayStart(resolutionPolicy);
    const activeAdapter = getActiveLoRAIdentityForPipeline(pipeline);
    assertSupportedGenerationOptions(options);
    const generationConfig = resolveGenerationConfigEvidence(pipeline, options);
    const result = await pipeline.generateTokenIds(prompt, options);
    const tokenIds = Array.from(result?.tokenIds || [], Number);
    const outputText = decodeGeneratedTokens(pipeline, tokenIds);
    const stats = result?.stats || pipeline.getStats?.() || null;
    const kernelCapabilities = typeof pipeline.getKernelCapabilities === 'function'
      ? pipeline.getKernelCapabilities()
      : getKernelCapabilities();
    const backendIdentity = buildGenerationBackendIdentity({
      deviceInfo: kernelCapabilities?.adapterInfo || null,
      kernelCapabilities,
      stats,
    });
    assertActiveAdapterUnchanged(pipeline, activeAdapter);
    return buildGenerationEvidence({
      outputText,
      tokenIds,
      generationConfig,
      logicalModelId: resolved.logicalModelId ?? resolved.modelId,
      modelId: resolved.modelId,
      manifestHash: resolved.manifestHash || null,
      resolvedRuntimeSessionId: pipeline.resolvedRuntimeSession?.id ?? null,
      activeAdapter,
      backendIdentity,
      stats,
      resolutionPolicy,
    });
  }

  async function embedWithEvidence(prompt, options = {}) {
    assertExecutionMayStart(resolutionPolicy);
    const activeAdapter = getActiveLoRAIdentityForPipeline(pipeline);
    const result = await pipeline.embed(prompt, options);
    const stats = pipeline.getStats?.() || null;
    const kernelCapabilities = typeof pipeline.getKernelCapabilities === 'function'
      ? pipeline.getKernelCapabilities()
      : getKernelCapabilities();
    const backendIdentity = buildGenerationBackendIdentity({
      deviceInfo: kernelCapabilities?.adapterInfo || null,
      kernelCapabilities,
      stats,
    });
    assertActiveAdapterUnchanged(pipeline, activeAdapter);
    return buildEmbeddingEvidence({
      prompt,
      result,
      logicalModelId: resolved.logicalModelId ?? resolved.modelId,
      modelId: resolved.modelId,
      manifestHash: resolved.manifestHash || null,
      resolvedRuntimeSessionId: pipeline.resolvedRuntimeSession?.id ?? null,
      activeAdapter,
      backendIdentity,
      stats,
      resolutionPolicy,
    });
  }

  async function rerankWithEvidence(query, documents, options = {}) {
    assertExecutionMayStart(resolutionPolicy);
    const activeAdapter = getActiveLoRAIdentityForPipeline(pipeline);
    if (pipeline.manifest?.inference?.supportsRerank !== true) {
      throw new Error('Loaded Doppler manifest does not declare rerank support.');
    }
    const normalizedQuery = String(query || '').trim();
    if (!normalizedQuery) {
      throw new Error('Doppler rerankWithEvidence requires a non-empty query.');
    }
    if (!Array.isArray(documents) || documents.length === 0) {
      throw new Error('Doppler rerankWithEvidence requires a non-empty documents array.');
    }
    const normalizedDocuments = documents.map((document, index) => {
      const text = String(document || '').trim();
      if (!text) throw new Error(`Doppler rerank document ${index} must be non-empty.`);
      return text;
    });
    const {
      resolveRerankScoringConfig,
      scoreRerankDocument,
    } = await import('../../inference/rerank.js');
    const scoringConfig = resolveRerankScoringConfig(pipeline);
    const scores = [];
    for (let index = 0; index < normalizedDocuments.length; index += 1) {
      const scored = await scoreRerankDocument(
        pipeline,
        normalizedQuery,
        normalizedDocuments[index],
        scoringConfig,
        { benchmark: options.benchmark === true }
      );
      scores.push({
        index,
        document: normalizedDocuments[index],
        score: scored.score,
        probability: scored.probability,
        trueLogit: scored.trueLogit,
        falseLogit: scored.falseLogit,
        tokenCount: scored.tokenCount,
        scoringPath: scored.scoringPath,
      });
    }
    const ranking = [...scores]
      .sort((left, right) => (right.score - left.score) || (left.index - right.index))
      .map((entry, index) => ({ rank: index + 1, ...entry }));
    const stats = pipeline.getStats?.() || null;
    const kernelCapabilities = typeof pipeline.getKernelCapabilities === 'function'
      ? pipeline.getKernelCapabilities()
      : getKernelCapabilities();
    const backendIdentity = buildGenerationBackendIdentity({
      deviceInfo: kernelCapabilities?.adapterInfo || null,
      kernelCapabilities,
      stats,
    });
    assertActiveAdapterUnchanged(pipeline, activeAdapter);
    const identity = await buildResolutionIdentity({
      logicalModelId: resolved.logicalModelId ?? resolved.modelId,
      modelId: resolved.modelId,
      manifestHash: resolved.manifestHash || null,
      resolvedRuntimeSessionId: pipeline.resolvedRuntimeSession?.id ?? null,
      activeAdapter,
      backendIdentity,
      resolutionPolicy,
    });
    return {
      schema: RERANK_EVIDENCE_SCHEMA,
      query: normalizedQuery,
      documents: normalizedDocuments,
      scores,
      ranking,
      inputHash: hashEvidenceValue({ query: normalizedQuery, documents: normalizedDocuments }),
      outputHash: hashEvidenceValue({ scores, ranking }),
      resolution: identity.resolution,
      executionIdentity: identity.executionIdentity,
      backendIdentity,
      backendIdentityHash: hashEvidenceValue(backendIdentity),
      stats: stats && typeof stats === 'object' ? stats : null,
    };
  }

  const handle = {
    generate(prompt, options = {}) {
      assertRaw('generate');
      assertSupportedGenerationOptions(options);
      return pipeline.generate(prompt, options);
    },
    async generateText(prompt, options = {}) {
      assertRaw('generateText');
      assertSupportedGenerationOptions(options);
      return collectText(pipeline.generate(prompt, options));
    },
    generateWithEvidence,
    chat(messages, options = {}) {
      assertRaw('chat');
      assertSupportedGenerationOptions(options);
      return pipeline.generate(messages, options);
    },
    async chatText(messages, options = {}) {
      assertSupportedGenerationOptions(options);
      const evidence = await generateWithEvidence(messages, options);
      const content = evidence.outputText;
      const promptText = resolveChatPromptForUsage(pipeline, messages);
      const promptTokens = countTokens(pipeline, promptText);
      const completionTokens = evidence.tokenIds.length;
      return {
        content,
        usage: {
          promptTokens,
          completionTokens,
          totalTokens: promptTokens + completionTokens,
        },
        evidence,
      };
    },
    async embed(prompt, options = {}) {
      assertRaw('embed');
      return pipeline.embed(prompt, options);
    },
    embedWithEvidence,
    async embedBatch(prompts, options = {}) {
      assertRaw('embedBatch');
      return pipeline.embedBatch(prompts, options);
    },
    rerankWithEvidence,
    async encodeSequence(sequence, options = {}) {
      assertRaw('encodeSequence');
      return pipeline.encodeSequence(sequence, options);
    },
    async embedImage(args = {}) {
      assertRaw('embedImage');
      return pipeline.embedImage(args);
    },
    async embedAudio(args = {}) {
      assertRaw('embedAudio');
      return pipeline.embedAudio(args);
    },
    async transcribeImage(args = {}) {
      assertRaw('transcribeImage');
      return pipeline.transcribeImage(args);
    },
    async transcribeAudio(args = {}) {
      assertRaw('transcribeAudio');
      return pipeline.transcribeAudio(args);
    },
    async transcribeVideo(args = {}) {
      assertRaw('transcribeVideo');
      return pipeline.transcribeVideo(args);
    },
    get supportsEmbedding() {
      return pipeline.manifest?.modelType === 'embedding'
        || pipeline.manifest?.inference?.supportsEmbedding === true;
    },
    get supportsRerank() {
      return pipeline.manifest?.inference?.supportsRerank === true;
    },
    get supportsSequence() {
      return pipeline.manifest?.inference?.supportsSequence === true;
    },
    get supportsTranscription() {
      return pipeline.manifest?.inference?.supportsTranscription === true
        && pipeline.audioCapable === true;
    },
    get supportsVision() {
      return pipeline.manifest?.inference?.supportsVision === true
        && pipeline.visionCapable === true;
    },
    async loadLoRA(adapter, loadOptions = {}) {
      return loadLoRAAdapterForPipeline(pipeline, adapter, loadOptions);
    },
    async activateLoRAFromTrainingOutput(trainingOutput) {
      return activateLoRAFromTrainingOutputForPipeline(pipeline, trainingOutput);
    },
    async unloadLoRA() {
      return unloadLoRAAdapterForPipeline(pipeline);
    },
    resetGenerationState() {
      if (typeof pipeline.resetGenerationState === 'function') {
        return pipeline.resetGenerationState();
      }
      if (typeof pipeline.resetToSeqLen === 'function') {
        return pipeline.resetToSeqLen(0);
      }
      throw new Error('Loaded Doppler pipeline does not expose generation-state reset');
    },
    async unload() {
      await pipeline.unload();
    },
    get activeLoRA() {
      return getActiveLoRAForPipeline(pipeline);
    },
    get loaded() {
      return pipeline.isLoaded === true;
    },
    get modelId() {
      return resolved.modelId;
    },
    get logicalModelId() {
      return resolved.logicalModelId ?? resolved.modelId;
    },
    get resolvedArtifactVariantId() {
      return resolved.manifestHash
        ? normalizeSha256Identity(resolved.manifestHash, 'manifestHash')
        : null;
    },
    get resolutionPolicy() {
      return resolutionPolicy;
    },
    get manifestHash() {
      return resolved.manifestHash || null;
    },
    get persistentCache() {
      return resolved.persistentCache || null;
    },
    get manifest() {
      return pipeline.manifest;
    },
    get deviceInfo() {
      return getKernelCapabilities()?.adapterInfo ?? null;
    },
    advanced: {
      tokenizeText(text) {
        return tokenizeText(pipeline, text);
      },
      tokenizePrompt(prompt, options = {}) {
        return tokenizePrompt(pipeline, prompt, options);
      },
      decodeTokenIds(tokenIds) {
        if (!Array.isArray(tokenIds)) {
          throw new Error('Doppler advanced.decodeTokenIds requires an array.');
        }
        return String(pipeline.tokenizer.decode(tokenIds, true, false));
      },
      getSpecialTokens() {
        return pipeline.tokenizer?.getSpecialTokens?.() ?? {};
      },
      getStopTokenIds() {
        return Array.isArray(pipeline.modelConfig?.stopTokenIds)
          ? [...pipeline.modelConfig.stopTokenIds]
          : [];
      },
      getStats() {
        return pipeline.getStats?.() ?? null;
      },
      prefillKV(prompt, options = {}) {
        assertRaw('advanced.prefillKV');
        assertSupportedGenerationOptions(options);
        return pipeline.prefillKVOnly(prompt, options);
      },
      resetToSeqLen(seqLen) {
        return pipeline.resetToSeqLen(seqLen);
      },
      prefillWithLogits(prompt, options = {}) {
        assertRaw('advanced.prefillWithLogits');
        assertSupportedGenerationOptions(options);
        return pipeline.prefillWithLogits(prompt, options);
      },
      prefillWithTokenLogits(prompt, tokenIds, options = {}) {
        assertRaw('advanced.prefillWithTokenLogits');
        assertSupportedGenerationOptions(options);
        return pipeline.prefillWithTokenLogits(prompt, tokenIds, options);
      },
      prefillWithTokenLogitsFromKV(prefix, prompt, tokenIds, options = {}) {
        assertRaw('advanced.prefillWithTokenLogitsFromKV');
        assertSupportedGenerationOptions(options);
        return pipeline.prefillWithTokenLogitsFromKV(prefix, prompt, tokenIds, options);
      },
      decodeStepLogits(currentIds, options = {}) {
        assertRaw('advanced.decodeStepLogits');
        assertSupportedGenerationOptions(options);
        return pipeline.decodeStepLogits(currentIds, options);
      },
      generateWithPrefixKV(prefix, prompt, options = {}) {
        assertRaw('advanced.generateWithPrefixKV');
        assertSupportedGenerationOptions(options);
        return pipeline.generateWithPrefixKV(prefix, prompt, options);
      },
    },
  };

  handle.inspect = {
    listPolicies() {
      return listObservationPolicies();
    },
    async generate(prompt, options = {}) {
      if (typeof prompt !== 'string' || !prompt.trim()) {
        throw new Error('Doppler model.inspect.generate requires a non-empty string prompt.');
      }
      const policy = resolveObservationPolicy(options.policyId);
      const generationOptions = resolveInspectionGenerationOptions(options, policy);
      assertSupportedGenerationOptions(generationOptions);
      const logitsByStep = [];
      if (policy.requiredCaptures.includes('selected-token-probabilities')) {
        generationOptions.onLogits = (logits) => {
          logitsByStep.push(Float32Array.from(logits));
        };
      }
      const startedAt = performance.now();
      const evidence = await generateWithEvidence(prompt, generationOptions);
      const completedAt = performance.now();
      const promptTokenIds = tokenizeText(pipeline, prompt);
      const tokenRecords = policy.perplexity
        ? buildInspectionTokenRecords(
          evidence.tokenIds,
          logitsByStep,
          pipeline.tokenizer,
          Number.isInteger(options.topKSize) ? options.topKSize : 5
        )
        : [];
      const quality = policy.perplexity
        ? aggregateWordPerplexity(tokenRecords, {
          windowUnit: policy.perplexity.rollingWindow.unit,
          windowSize: policy.perplexity.rollingWindow.size,
        })
        : null;
      const fingerprint = buildComparisonFingerprint({
        artifact: {
          modelId: resolved.modelId,
          manifestHash: resolved.manifestHash,
        },
        tokenizer: resolveTokenizerContract(pipeline),
        promptTokenIds,
        sampling: evidence.generationConfig,
        observationPolicyId: policy.id,
        execution: evidence.backendIdentity,
        browser: resolveInspectionBrowserIdentity(),
        adapter: evidence.backendIdentity.adapter,
      });
      const receipt = {
        schema: MODEL_INSPECTION_RECEIPT_SCHEMA,
        policy,
        fingerprint,
        outputText: evidence.outputText,
        generatedTokenIds: [...evidence.tokenIds],
        wallTimingMs: completedAt - startedAt,
        performanceRepresentative: policy.performanceRepresentative,
        tokens: tokenRecords,
        quality,
        generationEvidence: evidence,
      };
      if (typeof options.onEvent === 'function') {
        options.onEvent({ type: 'inspection-complete', receipt });
      }
      return receipt;
    },
  };

  return handle;
}
