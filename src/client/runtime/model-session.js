import { getKernelCapabilities } from '../../gpu/device.js';
import { formatChatMessages } from '../../inference/pipelines/text/chat-format.js';
import { resolveSamplingConfig } from '../../inference/pipelines/text/sampling-config.js';
import { DOPPLER_VERSION } from '../../version.js';
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
  loadLoRAAdapterForPipeline,
  unloadLoRAAdapterForPipeline,
} from './lora.js';

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
const RUNTIME_PROFILE_SCHEMA = 'doppler_runtime_profile/v1';

function canonicalizeEvidence(value) {
  if (value === null || typeof value === 'boolean' || typeof value === 'string') {
    return JSON.stringify(value);
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new Error('Doppler generation evidence cannot hash non-finite numbers.');
    }
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((entry) => canonicalizeEvidence(entry)).join(',')}]`;
  }
  if (value && typeof value === 'object') {
    const entries = Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalizeEvidence(value[key])}`);
    return `{${entries.join(',')}}`;
  }
  throw new Error(`Doppler generation evidence cannot hash ${typeof value} values.`);
}

function bytesToHex(bytes) {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
}

async function hashEvidenceValue(value) {
  if (!globalThis.crypto?.subtle) {
    throw new Error('Doppler generation evidence requires Web Crypto SHA-256 support.');
  }
  const bytes = new TextEncoder().encode(canonicalizeEvidence(value));
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
  return `sha256:${bytesToHex(new Uint8Array(digest))}`;
}

function cleanEvidenceString(value) {
  const text = String(value ?? '').trim();
  return text || null;
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

async function buildGenerationEvidence({
  outputText,
  tokenIds,
  generationConfig,
  modelId,
  manifestHash,
  activeAdapter,
  backendIdentity,
  stats,
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
  const runtimeProfile = {
    schema: RUNTIME_PROFILE_SCHEMA,
    runtime: {
      package: 'doppler-gpu',
      version: DOPPLER_VERSION,
      surface: isNodeRuntime() ? 'node' : 'browser',
    },
    model: {
      modelId: cleanEvidenceString(modelId),
      manifestHash: cleanEvidenceString(manifestHash),
      activeAdapter: cleanEvidenceString(activeAdapter),
    },
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
    runtimeProfile,
    runtimeProfileHash,
    backendIdentity,
    backendIdentityHash,
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

export function createModelHandle(pipeline, resolved) {
  async function generateWithEvidence(prompt, options = {}) {
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
    return buildGenerationEvidence({
      outputText,
      tokenIds,
      generationConfig,
      modelId: resolved.modelId,
      manifestHash: resolved.manifestHash || null,
      activeAdapter: getActiveLoRAForPipeline(pipeline),
      backendIdentity,
      stats,
    });
  }

  const handle = {
    generate(prompt, options = {}) {
      assertSupportedGenerationOptions(options);
      return pipeline.generate(prompt, options);
    },
    async generateText(prompt, options = {}) {
      assertSupportedGenerationOptions(options);
      return collectText(pipeline.generate(prompt, options));
    },
    generateWithEvidence,
    chat(messages, options = {}) {
      assertSupportedGenerationOptions(options);
      return pipeline.generate(messages, options);
    },
    async chatText(messages, options = {}) {
      assertSupportedGenerationOptions(options);
      const content = await collectText(pipeline.generate(messages, options));
      const promptText = resolveChatPromptForUsage(pipeline, messages);
      const promptTokens = countTokens(pipeline, promptText);
      const completionTokens = countTokens(pipeline, content);
      return {
        content,
        usage: {
          promptTokens,
          completionTokens,
          totalTokens: promptTokens + completionTokens,
        },
      };
    },
    async embed(prompt, options = {}) {
      return pipeline.embed(prompt, options);
    },
    async embedBatch(prompts, options = {}) {
      return pipeline.embedBatch(prompts, options);
    },
    async encodeSequence(sequence, options = {}) {
      return pipeline.encodeSequence(sequence, options);
    },
    async embedImage(args = {}) {
      return pipeline.embedImage(args);
    },
    async embedAudio(args = {}) {
      return pipeline.embedAudio(args);
    },
    async transcribeImage(args = {}) {
      return pipeline.transcribeImage(args);
    },
    async transcribeAudio(args = {}) {
      return pipeline.transcribeAudio(args);
    },
    async transcribeVideo(args = {}) {
      return pipeline.transcribeVideo(args);
    },
    get supportsEmbedding() {
      return pipeline.manifest?.modelType === 'embedding'
        || pipeline.manifest?.inference?.supportsEmbedding === true;
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
      prefillKV(prompt, options = {}) {
        assertSupportedGenerationOptions(options);
        return pipeline.prefillKVOnly(prompt, options);
      },
      resetToSeqLen(seqLen) {
        return pipeline.resetToSeqLen(seqLen);
      },
      prefillWithLogits(prompt, options = {}) {
        assertSupportedGenerationOptions(options);
        return pipeline.prefillWithLogits(prompt, options);
      },
      prefillWithTokenLogits(prompt, tokenIds, options = {}) {
        assertSupportedGenerationOptions(options);
        return pipeline.prefillWithTokenLogits(prompt, tokenIds, options);
      },
      prefillWithTokenLogitsFromKV(prefix, prompt, tokenIds, options = {}) {
        assertSupportedGenerationOptions(options);
        return pipeline.prefillWithTokenLogitsFromKV(prefix, prompt, tokenIds, options);
      },
      decodeStepLogits(currentIds, options = {}) {
        assertSupportedGenerationOptions(options);
        return pipeline.decodeStepLogits(currentIds, options);
      },
      generateWithPrefixKV(prefix, prompt, options = {}) {
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
