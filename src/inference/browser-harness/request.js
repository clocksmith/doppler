import { sha256Hex } from '../../formats/sha256.js';
import { stableSortObject } from '../../formats/stable-sort-object.js';

export const BROWSER_WORKLOAD_SET = Object.freeze([
  'kernels',
  'inference',
  'embedding',
  'rerank',
  'training',
  'diffusion',
  'energy',
]);

export const BROWSER_MODE_SET = Object.freeze(['verify', 'debug', 'bench', 'diagnose']);

export const BROWSER_WORKLOAD_DISPATCH_MAP = Object.freeze({
  verify: Object.freeze({
    kernels: 'runKernelSuite',
    inference: 'runInferenceSuite',
    embedding: 'runEmbeddingSuite',
    rerank: 'runRerankSuite',
    training: 'runTrainingSuite',
    diffusion: 'runDiffusionSuite',
    energy: 'runEnergySuite',
  }),
  debug: Object.freeze({
    inference: 'runInferenceSuite(debug)',
    embedding: 'runEmbeddingSuite(debug)',
  }),
  diagnose: Object.freeze({
    inference: 'runInferenceSuite(diagnose)',
  }),
  bench: Object.freeze({
    inference: 'runBenchSuite',
    embedding: 'runBenchSuite',
    rerank: 'runBenchSuite',
    training: 'runBenchSuite(training)',
    diffusion: 'runBenchSuite(diffusion)',
  }),
});

export function isPlainObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

export function resolveExecutionPlanCadence(executionPlan) {
  if (!isPlainObject(executionPlan)) {
    return null;
  }
  const finalActivePlanId = typeof executionPlan.finalActivePlanId === 'string'
    ? executionPlan.finalActivePlanId
    : null;
  for (const candidate of [executionPlan.primary, executionPlan.fallback]) {
    if (!isPlainObject(candidate)) {
      continue;
    }
    if (finalActivePlanId == null || candidate.id === finalActivePlanId) {
      return candidate;
    }
  }
  return isPlainObject(executionPlan.primary) ? executionPlan.primary : null;
}

export function resolveDecodeCadence(runtimeConfig, executionPlan = null) {
  const inference = runtimeConfig?.inference;
  const batching = inference?.batching;
  const session = inference?.session;
  const decodeLoop = session?.decodeLoop;
  if (!isPlainObject(batching) || !isPlainObject(decodeLoop)) {
    return null;
  }
  const executionPlanCadence = resolveExecutionPlanCadence(executionPlan);
  const batchSize = executionPlanCadence?.batchSize ?? decodeLoop.batchSize ?? batching.batchSize ?? null;
  const readbackInterval = executionPlanCadence?.readbackInterval ?? decodeLoop.readbackInterval ?? batching.readbackInterval ?? null;
  const maxBatchDecodeTokens = executionPlanCadence?.maxBatchDecodeTokens
    ?? decodeLoop.maxBatchDecodeTokens
    ?? null;
  return {
    batchSize,
    readbackInterval,
    maxBatchDecodeTokens,
    stopCheckMode: executionPlanCadence?.stopCheckMode ?? decodeLoop.stopCheckMode ?? batching.stopCheckMode ?? null,
    readbackMode: executionPlanCadence?.readbackMode ?? decodeLoop.readbackMode ?? batching.readbackMode ?? null,
    disableCommandBatching: executionPlanCadence?.disableCommandBatching ?? decodeLoop.disableCommandBatching ?? null,
    disableMultiTokenDecode: inference?.generation?.disableMultiTokenDecode === true,
    speculationMode: session?.speculation?.mode ?? null,
    tokensPerReadback: Number.isFinite(batchSize) && Number.isFinite(readbackInterval)
      ? batchSize * readbackInterval
      : null,
    runtimeMirror: {
      batching: {
        batchSize: batching.batchSize ?? null,
        readbackInterval: batching.readbackInterval ?? null,
        stopCheckMode: batching.stopCheckMode ?? null,
        readbackMode: batching.readbackMode ?? null,
      },
      decodeLoop: {
        batchSize: decodeLoop.batchSize ?? null,
        readbackInterval: decodeLoop.readbackInterval ?? null,
        maxBatchDecodeTokens: decodeLoop.maxBatchDecodeTokens ?? null,
        stopCheckMode: decodeLoop.stopCheckMode ?? null,
        readbackMode: decodeLoop.readbackMode ?? null,
        ringTokens: decodeLoop.ringTokens ?? null,
        ringStop: decodeLoop.ringStop ?? null,
        ringStaging: decodeLoop.ringStaging ?? null,
      },
    },
    executionPlan: executionPlanCadence
      ? {
        id: executionPlanCadence.id ?? null,
        batchSize: executionPlanCadence.batchSize ?? null,
        readbackInterval: executionPlanCadence.readbackInterval ?? null,
        maxBatchDecodeTokens: executionPlanCadence.maxBatchDecodeTokens ?? null,
        stopCheckMode: executionPlanCadence.stopCheckMode ?? null,
        readbackMode: executionPlanCadence.readbackMode ?? null,
        disableCommandBatching: executionPlanCadence.disableCommandBatching ?? null,
        ringTokens: executionPlanCadence.ringTokens ?? null,
        ringStop: executionPlanCadence.ringStop ?? null,
        ringStaging: executionPlanCadence.ringStaging ?? null,
      }
      : null,
  };
}

export function getBrowserSupportedSuites() {
  return [...BROWSER_WORKLOAD_SET];
}

export function getBrowserSuiteDispatchMap() {
  return {
    verify: { ...BROWSER_WORKLOAD_DISPATCH_MAP.verify },
    debug: { ...BROWSER_WORKLOAD_DISPATCH_MAP.debug },
    bench: { ...BROWSER_WORKLOAD_DISPATCH_MAP.bench },
  };
}

export function getAllowedWorkloadsForMode(mode) {
  return Object.keys(BROWSER_WORKLOAD_DISPATCH_MAP[mode] || {});
}

export function createUnsupportedWorkloadError(requestedWorkload, context = {}) {
  const command = typeof context.command === 'string' && context.command.trim()
    ? context.command.trim()
    : 'run-browser-suite';
  const surface = typeof context.surface === 'string' && context.surface.trim()
    ? context.surface.trim()
    : 'browser';
  const mode = typeof context.mode === 'string' && context.mode.trim()
    ? context.mode.trim()
    : 'verify';
  const allowedWorkloads = getAllowedWorkloadsForMode(mode);
  const error = new Error(
    `Unsupported workload "${requestedWorkload}". Allowed workloads: ${allowedWorkloads.join(', ')}. ` +
    `command="${command}" mode="${mode}" surface="${surface}".`
  );
  error.code = 'unsupported_workload';
  error.requestedWorkload = requestedWorkload;
  error.allowedWorkloads = allowedWorkloads;
  error.command = command;
  error.mode = mode;
  error.surface = surface;
  error.details = {
    requestedWorkload,
    allowedWorkloads,
    command,
    mode,
    surface,
  };
  return error;
}

export function resolveHarnessContext(options = {}) {
  const command = typeof options.command === 'string' ? options.command : null;
  const surface = typeof options.surface === 'string' ? options.surface : null;
  const mode = typeof options.mode === 'string' ? options.mode : null;
  return {
    command: command ?? 'run-browser-suite',
    mode: mode ?? command ?? 'verify',
    surface: surface ?? 'browser',
  };
}

export function normalizeLegacySuite(value) {
  const suite = String(value || '').trim().toLowerCase();
  if (!suite) {
    return null;
  }
  return suite === 'benchmark' ? 'bench' : suite;
}

export function normalizeMode(value, context = {}) {
  const mode = String(value || '').trim().toLowerCase();
  if (!mode) {
    return 'verify';
  }
  if (!BROWSER_MODE_SET.includes(mode)) {
    throw new Error(`Unsupported browser harness mode "${mode}" for command "${context.command || 'run-browser-suite'}".`);
  }
  return mode;
}

export function resolveHarnessMode(options = {}, context = {}) {
  const explicitMode = options.mode ?? options.command ?? null;
  if (explicitMode) {
    return normalizeMode(explicitMode, context);
  }
  const legacySuite = normalizeLegacySuite(options.suite);
  if (legacySuite === 'debug' || legacySuite === 'bench') {
    return legacySuite;
  }
  return 'verify';
}

export function normalizeWorkload(value, mode, context = {}) {
  const workload = String(value || '').trim().toLowerCase();
  if (!workload) {
    throw createUnsupportedWorkloadError(workload, { ...context, mode });
  }
  if (!getAllowedWorkloadsForMode(mode).includes(workload)) {
    throw createUnsupportedWorkloadError(workload, { ...context, mode });
  }
  return workload;
}

export function resolveWorkload(options = {}, mode, context = {}) {
  if (options.workload) {
    return normalizeWorkload(options.workload, mode, context);
  }
  const legacySuite = normalizeLegacySuite(options.suite);
  if (legacySuite && legacySuite !== 'debug' && legacySuite !== 'bench') {
    return normalizeWorkload(legacySuite, mode, context);
  }
  if (mode === 'debug' || mode === 'bench') {
    return 'inference';
  }
  return normalizeWorkload('', mode, context);
}

export function resolveDispatchSuite(mode, workload) {
  if (mode === 'debug') {
    return 'debug';
  }
  if (mode === 'bench') {
    return 'bench';
  }
  return workload;
}

export function stableJson(value) {
  return JSON.stringify(stableSortObject(value)) ?? 'null';
}

export function hashStableJson(value) {
  return `sha256:${sha256Hex(stableJson(value))}`;
}

export function resolveExecutionGraphHash(manifest) {
  const execution = manifest?.inference?.execution;
  if (!execution || typeof execution !== 'object') {
    return null;
  }
  return hashStableJson(execution);
}
