import { log } from '../debug/index.js';
import { ERROR_CODES, createDopplerError } from '../errors/doppler-error.js';
import { createDopplerRuntimeService } from './runtime/index.js';
import { classifyProviderFailure } from './failure-taxonomy.js';
import { buildProviderReceiptV1 } from './receipt.js';
import { createFaultInjector } from './fault-injection.js';

// Canonical four routing modes (30-hybrid-routing-sdk.md).
const VALID_POLICY_MODES = new Set([
  'local-only',
  'prefer-local',
  'prefer-cloud',
  'cloud-only',
]);

function isPolicyFallbackEligible(failureClass, fallbackOn) {
  if (!Array.isArray(fallbackOn) || fallbackOn.length === 0) {
    return true;
  }
  return fallbackOn.includes(failureClass);
}

function safeGetDeviceSnapshot() {
  try {
    // Dynamic import avoided — these are pure reads of cached state.
    // In worker contexts, getKernelCapabilities / getDeviceEpoch may not be
    // available. Receipt will have device: null in that case.
    return { kernelCapabilities: null, deviceEpoch: 0 };
  } catch {
    return { kernelCapabilities: null, deviceEpoch: 0 };
  }
}

function validateProviderConfig(config) {
  if (!config || typeof config !== 'object') {
    throw new Error('createDopplerProvider requires a config object.');
  }

  const policyMode = config.policy?.mode || 'prefer-local';
  if (!VALID_POLICY_MODES.has(policyMode)) {
    throw new Error(`Invalid policy mode "${policyMode}". Must be one of: ${[...VALID_POLICY_MODES].join(', ')}.`);
  }

  const cloudOnlyMode = policyMode === 'cloud-only';
  if (!cloudOnlyMode && (!config.local || (!config.local.model && !config.local.handle))) {
    throw new Error('Provider config requires local.model or local.handle when policy mode is not "cloud-only".');
  }

  if (policyMode !== 'local-only' && config.fallback) {
    if (!config.fallback.provider) {
      throw new Error('Provider config fallback requires fallback.provider.');
    }
    if (!config.fallback.model) {
      throw new Error('Provider config fallback requires fallback.model.');
    }
  }
}

async function callOpenAIFallback(config, prompt, options) {
  const fallback = config.fallback;
  const baseUrl = String(fallback.baseUrl || 'https://api.openai.com').replace(/\/+$/, '');
  const apiKey = fallback.apiKey || null;

  const messages = Array.isArray(prompt)
    ? prompt
    : [{ role: 'user', content: String(prompt) }];

  const body = {
    model: fallback.model,
    messages,
    temperature: options.temperature ?? undefined,
    max_tokens: options.maxTokens ?? undefined,
  };

  const headers = { 'Content-Type': 'application/json' };
  if (apiKey) {
    headers['Authorization'] = `Bearer ${apiKey}`;
  }

  let response;
  try {
    response = await fetch(`${baseUrl}/v1/chat/completions`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    });
  } catch (fetchError) {
    throw createDopplerError(
      ERROR_CODES.PROVIDER_NETWORK_FAILED,
      `Fallback provider network error: ${fetchError instanceof Error ? fetchError.message : String(fetchError)}`
    );
  }

  if (!response.ok) {
    const errorText = await response.text().catch(() => '');
    throw createDopplerError(
      ERROR_CODES.PROVIDER_FALLBACK_FAILED,
      `Fallback provider returned ${response.status}: ${errorText}`
    );
  }

  const result = await response.json();
  const choice = result?.choices?.[0];
  if (!choice) {
    throw createDopplerError(
      ERROR_CODES.PROVIDER_FALLBACK_FAILED,
      'Fallback provider returned no choices.'
    );
  }

  return String(choice.message?.content ?? '');
}

function isNodeRuntime() {
  return typeof process !== 'undefined'
    && typeof process.versions === 'object'
    && typeof process.versions.node === 'string';
}

async function ensureWebGPUAvailable() {
  if (typeof globalThis.navigator !== 'undefined' && globalThis.navigator?.gpu) {
    return;
  }
  if (isNodeRuntime()) {
    const { bootstrapNodeWebGPU } = await import('../tooling/node-webgpu.js');
    const result = await bootstrapNodeWebGPU();
    if (result.ok && globalThis.navigator?.gpu) {
      return;
    }
  }
  throw createDopplerError(
    ERROR_CODES.GPU_UNAVAILABLE,
    'WebGPU is unavailable. Install a Node WebGPU provider or run in a WebGPU-capable browser.'
  );
}

export function createDopplerProvider(config) {
  validateProviderConfig(config);

  const policyMode = config.policy?.mode || 'prefer-local';
  const policyId = config.policy?.id || null;
  const fallbackOn = config.policy?.fallbackOn || null;
  const emitReceipts = config.diagnostics?.receipts !== false;
  const injector = createFaultInjector(config);

  const hasPreloadedHandle = Boolean(config.local?.handle);
  const runtime = hasPreloadedHandle ? null : createDopplerRuntimeService({ ensureWebGPUAvailable });
  let localModelHandle = hasPreloadedHandle ? config.local.handle : null;
  let localModelLoading = null;

  async function ensureLocalModel() {
    if (localModelHandle?.loaded) {
      return localModelHandle;
    }
    if (localModelLoading) {
      return localModelLoading;
    }
    if (!runtime) {
      throw new Error('No runtime available and pre-loaded handle is not loaded.');
    }
    localModelLoading = runtime.load(config.local.model, {
      runtimeConfig: config.local.runtimeConfig,
      onProgress: config.local.onProgress,
    }).then((handle) => {
      localModelHandle = handle;
      localModelLoading = null;
      return handle;
    }).catch((error) => {
      localModelLoading = null;
      throw error;
    });
    return localModelLoading;
  }

  async function generateLocal(prompt, options) {
    const handle = await ensureLocalModel();
    const text = await handle.generateText(prompt, options);
    return {
      text,
      modelId: handle.modelId,
      modelHash: handle.manifest?.meta?.hash ?? null,
      device: handle.deviceInfo,
    };
  }

  function makeReceipt(params) {
    if (!emitReceipts) return null;
    const { kernelCapabilities, deviceEpoch } = safeGetDeviceSnapshot();
    return buildProviderReceiptV1({
      ...params,
      policyMode,
      policyId,
      kernelCapabilities,
      deviceEpoch,
    });
  }

  async function generate(promptOrOptions) {
    const { prompt, ...options } = typeof promptOrOptions === 'string'
      ? { prompt: promptOrOptions }
      : promptOrOptions;

    const startTime = Date.now();

    // cloud-only: never try local.
    if (policyMode === 'cloud-only') {
      if (!config.fallback) {
        throw createDopplerError(ERROR_CODES.PROVIDER_FALLBACK_NOT_CONFIGURED, 'Policy is cloud-only but no fallback is configured.');
      }
      const fallbackStart = Date.now();
      const text = await callOpenAIFallback(config, prompt, options);
      const receipt = makeReceipt({
        source: 'fallback',
        model: { id: config.fallback.model, fallbackId: config.fallback.model },
        fallbackDecision: { reason: 'policy_cloud_only', eligible: true, executed: true, deniedReason: null },
        fallbackDurationMs: Date.now() - fallbackStart,
        totalDurationMs: Date.now() - startTime,
      });
      return { text, inferenceSource: 'fallback', receipt };
    }

    // prefer-cloud: try cloud first, fall back to local on cloud failure.
    if (policyMode === 'prefer-cloud') {
      if (!config.fallback) {
        throw createDopplerError(ERROR_CODES.PROVIDER_FALLBACK_NOT_CONFIGURED, 'Policy is prefer-cloud but no fallback is configured.');
      }
      const cloudStart = Date.now();
      try {
        const text = await callOpenAIFallback(config, prompt, options);
        const receipt = makeReceipt({
          source: 'fallback',
          model: { id: config.fallback.model, fallbackId: config.fallback.model },
          fallbackDecision: { reason: 'policy_prefer_cloud', eligible: true, executed: true, deniedReason: null },
          fallbackDurationMs: Date.now() - cloudStart,
          totalDurationMs: Date.now() - startTime,
        });
        return { text, inferenceSource: 'fallback', receipt };
      } catch (cloudError) {
        const cloudFailure = classifyProviderFailure(cloudError, { surface: 'openai_compat' });
        log.debug('provider', `Cloud inference failed: [${cloudFailure.failureClass}] ${cloudError instanceof Error ? cloudError.message : String(cloudError)}`);
        // Fall through to local.
        const localStart = Date.now();
        try {
          const result = await generateLocal(prompt, options);
          const receipt = makeReceipt({
            source: 'local',
            model: { id: result.modelId, hash: result.modelHash },
            deviceInfo: result.device,
            failure: cloudFailure,
            fallbackDecision: { reason: cloudFailure.failureClass, eligible: true, executed: true, deniedReason: null },
            localDurationMs: Date.now() - localStart,
            fallbackDurationMs: Date.now() - cloudStart,
            totalDurationMs: Date.now() - startTime,
          });
          return { text: result.text, inferenceSource: 'local', receipt };
        } catch (localError) {
          const localFailure = classifyProviderFailure(localError);
          const receipt = makeReceipt({
            source: 'local',
            model: { id: config.local?.model || localModelHandle?.modelId || '' },
            failure: localFailure,
            fallbackDecision: { reason: 'both_failed', eligible: true, executed: true, deniedReason: null },
            localDurationMs: Date.now() - localStart,
            fallbackDurationMs: Date.now() - cloudStart,
            totalDurationMs: Date.now() - startTime,
          });
          const wrappedError = createDopplerError(ERROR_CODES.PROVIDER_LOCAL_FAILED, `Both cloud and local inference failed. Cloud: ${cloudError?.message ?? 'unknown'}. Local: ${localError?.message ?? 'unknown'}`);
          wrappedError.receipt = receipt;
          throw wrappedError;
        }
      }
    }

    // local-only and prefer-local: try local first.

    // Check fault injection before local inference.
    if (injector.shouldInject('generate')) {
      const injectedError = injector.createInjectedError();
      log.debug('provider', `Fault injection triggered: ${injectedError.message}`);
      return handleLocalFailure(injectedError, prompt, options, startTime);
    }

    const localStart = Date.now();
    try {
      const result = await generateLocal(prompt, options);
      const receipt = makeReceipt({
        source: 'local',
        model: { id: result.modelId, hash: result.modelHash },
        deviceInfo: result.device,
        localDurationMs: Date.now() - localStart,
        totalDurationMs: Date.now() - startTime,
      });
      return { text: result.text, inferenceSource: 'local', receipt };
    } catch (error) {
      return handleLocalFailure(error, prompt, options, startTime, Date.now() - localStart);
    }
  }

  async function handleLocalFailure(localError, prompt, options, startTime, localDurationMs = null) {
    const failure = classifyProviderFailure(localError);
    const failureClass = failure.failureClass;
    log.debug('provider', `Local inference failed: [${failureClass}] ${localError instanceof Error ? localError.message : String(localError)}`);

    // Local failed. Check fallback policy.
    if (policyMode === 'local-only') {
      const receipt = makeReceipt({
        source: 'local',
        model: { id: config.local?.model || localModelHandle?.modelId || '' },
        failure,
        fallbackDecision: { reason: 'policy_local_only', eligible: false, executed: false, deniedReason: 'local-only policy' },
        localDurationMs,
        totalDurationMs: Date.now() - startTime,
      });
      const wrappedError = createDopplerError(ERROR_CODES.PROVIDER_LOCAL_FAILED, `Local inference failed and policy is local-only: ${localError?.message ?? 'unknown error'}`);
      wrappedError.receipt = receipt;
      throw wrappedError;
    }

    if (!config.fallback) {
      const receipt = makeReceipt({
        source: 'local',
        model: { id: config.local?.model || localModelHandle?.modelId || '' },
        failure,
        fallbackDecision: { reason: 'no_fallback_configured', eligible: false, executed: false, deniedReason: 'no fallback configured' },
        localDurationMs,
        totalDurationMs: Date.now() - startTime,
      });
      const wrappedError = createDopplerError(ERROR_CODES.PROVIDER_FALLBACK_NOT_CONFIGURED, `Local inference failed and no fallback is configured: ${localError?.message ?? 'unknown error'}`);
      wrappedError.receipt = receipt;
      throw wrappedError;
    }

    const eligible = isPolicyFallbackEligible(failureClass, fallbackOn);
    if (!eligible) {
      const receipt = makeReceipt({
        source: 'local',
        model: { id: config.local?.model || localModelHandle?.modelId || '' },
        failure,
        fallbackDecision: { reason: failureClass, eligible: false, executed: false, deniedReason: 'policy denied for this failure class' },
        localDurationMs,
        totalDurationMs: Date.now() - startTime,
      });
      const wrappedError = createDopplerError(ERROR_CODES.PROVIDER_POLICY_DENIED, `Local inference failed with ${failureClass} but policy does not allow fallback for this failure class.`);
      wrappedError.receipt = receipt;
      throw wrappedError;
    }

    // Fallback.
    log.debug('provider', `Falling back to ${config.fallback.provider}/${config.fallback.model} due to ${failureClass}`);
    const fallbackStart = Date.now();
    const text = await callOpenAIFallback(config, prompt, options);
    const receipt = makeReceipt({
      source: 'fallback',
      model: { id: config.fallback.model, fallbackId: config.fallback.model },
      failure,
      fallbackDecision: { reason: failureClass, eligible: true, executed: true, deniedReason: null },
      localDurationMs,
      fallbackDurationMs: Date.now() - fallbackStart,
      totalDurationMs: Date.now() - startTime,
    });
    return { text, inferenceSource: 'fallback', receipt };
  }

  async function unload() {
    if (localModelHandle) {
      await localModelHandle.unload();
      localModelHandle = null;
    }
  }

  return {
    generate,
    unload,
  };
}
