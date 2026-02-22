import { log } from '../../../debug/index.js';
import { resolveKernelPath } from '../../../config/kernel-path-loader.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';
import {
  resolveDeferredRoundingWindowTokens,
  resolveRangeAwareSelectiveWideningConfig,
} from './finiteness-policy.js';

export const PRIMARY_EXECUTION_PLAN_ID = 'primary';
export const FINITENESS_FALLBACK_EXECUTION_PLAN_ID = 'finiteness_fallback';

function normalizePositiveInt(value, fallback, label) {
  if (!Number.isFinite(value)) return fallback;
  const normalized = Math.floor(value);
  if (normalized >= 1) return normalized;
  log.warn('Pipeline', `[ExecutionPlan] ${label}=${value} is invalid; using ${fallback}.`);
  return fallback;
}

function normalizeStopCheckMode(value, fallback) {
  if (value === 'batch' || value === 'per-token') {
    return value;
  }
  return fallback;
}

function resolveFallbackActivationDtype(primaryActivationDtype) {
  const fallbackActivationDtype = selectRuleValue(
    'inference',
    'execution',
    'finitenessFallbackActivationDtype',
    { activationDtype: primaryActivationDtype }
  );
  if (fallbackActivationDtype !== 'f16' && fallbackActivationDtype !== 'f32') {
    throw new Error(
      `[ExecutionPlan] finiteness fallback activation dtype must be "f16" or "f32"; got "${fallbackActivationDtype}".`
    );
  }
  return fallbackActivationDtype;
}

function resolveFallbackKernelPath(primaryKernelPath) {
  const primaryKernelPathId = primaryKernelPath?.id ?? null;
  if (!primaryKernelPathId) {
    return {
      kernelPath: null,
      kernelPathId: null,
      kernelPathSource: 'none',
    };
  }

  const fallbackKernelPathId = selectRuleValue(
    'inference',
    'kernelPath',
    'finitenessFallback',
    { kernelPathId: primaryKernelPathId }
  );

  const resolvedKernelPathId = typeof fallbackKernelPathId === 'string' && fallbackKernelPathId.length > 0
    ? fallbackKernelPathId
    : primaryKernelPathId;

  try {
    const kernelPath = resolveKernelPath(resolvedKernelPathId);
    return {
      kernelPath,
      kernelPathId: resolvedKernelPathId,
      kernelPathSource: resolvedKernelPathId === primaryKernelPathId ? 'self' : 'rule',
    };
  } catch (error) {
    throw new Error(
      `[ExecutionPlan] Failed to resolve finiteness fallback kernel path "${resolvedKernelPathId}" ` +
      `(from "${primaryKernelPathId}"): ${error?.message || error}`
    );
  }
}

function createStaticExecutionPlan({
  id,
  source,
  kernelPath,
  kernelPathSource,
  activationDtype,
  finitenessPolicy,
  deferredRoundingWindowTokens,
  generationConfig,
  batchingConfig,
}) {
  return {
    id,
    source,
    kernelPath,
    kernelPathId: kernelPath?.id ?? null,
    kernelPathSource,
    activationDtype,
    finitenessGuardEnabled: activationDtype === 'f16' && finitenessPolicy.enabled,
    finitenessAbsThreshold: finitenessPolicy.absThreshold,
    finitenessIncludeNonFinite: finitenessPolicy.includeNonFinite,
    deferredRoundingWindowTokens,
    defaultDisableCommandBatching: generationConfig.disableCommandBatching,
    defaultDisableMultiTokenDecode: generationConfig.disableMultiTokenDecode,
    defaultBatchSize: batchingConfig.batchSize,
    defaultStopCheckMode: batchingConfig.stopCheckMode,
    defaultMaxTokens: batchingConfig.maxTokens,
    readbackInterval: batchingConfig.readbackInterval,
    ringTokens: batchingConfig.ringTokens,
    ringStop: batchingConfig.ringStop,
    ringStaging: batchingConfig.ringStaging,
  };
}

function getPlanState(container) {
  if (container?.executionPlanState) {
    return container.executionPlanState;
  }
  return container;
}

function getPlanById(planState, planId) {
  if (!planState) {
    throw new Error('[ExecutionPlan] plan state is not initialized.');
  }
  if (planId === PRIMARY_EXECUTION_PLAN_ID) {
    return planState.primaryPlan;
  }
  if (planId === FINITENESS_FALLBACK_EXECUTION_PLAN_ID) {
    return planState.fallbackPlan;
  }
  throw new Error(`[ExecutionPlan] unknown plan id "${planId}".`);
}

export function compileExecutionPlanState(options) {
  const runtimeConfig = options?.runtimeConfig;
  const resolvedKernelPath = options?.resolvedKernelPath ?? null;
  const kernelPathSource = options?.kernelPathSource ?? 'none';

  if (!runtimeConfig?.inference) {
    throw new Error('[ExecutionPlan] runtimeConfig.inference is required.');
  }

  const inferenceConfig = runtimeConfig.inference;
  const computeConfig = inferenceConfig.compute;
  const generationConfig = inferenceConfig.generation;
  const batchingConfig = inferenceConfig.batching;

  const finitenessPolicy = resolveRangeAwareSelectiveWideningConfig(computeConfig);
  const deferredRoundingWindowTokens = resolveDeferredRoundingWindowTokens(computeConfig);

  const primaryPlan = createStaticExecutionPlan({
    id: PRIMARY_EXECUTION_PLAN_ID,
    source: 'configured',
    kernelPath: resolvedKernelPath,
    kernelPathSource,
    activationDtype: computeConfig.activationDtype,
    finitenessPolicy,
    deferredRoundingWindowTokens,
    generationConfig,
    batchingConfig,
  });

  let fallbackPlan = null;
  if (primaryPlan.finitenessGuardEnabled) {
    const fallbackActivationDtype = resolveFallbackActivationDtype(primaryPlan.activationDtype);
    const fallbackKernelPathState = resolveFallbackKernelPath(primaryPlan.kernelPath);

    fallbackPlan = createStaticExecutionPlan({
      id: FINITENESS_FALLBACK_EXECUTION_PLAN_ID,
      source: 'finiteness-fallback',
      kernelPath: fallbackKernelPathState.kernelPath,
      kernelPathSource: fallbackKernelPathState.kernelPathSource,
      activationDtype: fallbackActivationDtype,
      finitenessPolicy,
      deferredRoundingWindowTokens,
      generationConfig,
      batchingConfig,
    });
  }

  return {
    primaryPlan,
    fallbackPlan,
    activePlanId: PRIMARY_EXECUTION_PLAN_ID,
  };
}

export function hasFallbackExecutionPlan(container) {
  const planState = getPlanState(container);
  return planState?.fallbackPlan != null;
}

export function resolveActiveExecutionPlan(container) {
  const planState = getPlanState(container);
  const activePlan = getPlanById(planState, planState?.activePlanId ?? PRIMARY_EXECUTION_PLAN_ID);
  if (!activePlan) {
    throw new Error('[ExecutionPlan] active plan is missing.');
  }
  return activePlan;
}

export function setActiveExecutionPlan(container, planId) {
  const planState = getPlanState(container);
  const plan = getPlanById(planState, planId);
  if (!plan) {
    throw new Error(`[ExecutionPlan] plan "${planId}" is not available.`);
  }
  planState.activePlanId = planId;
  return plan;
}

export function resetActiveExecutionPlan(container) {
  return setActiveExecutionPlan(container, PRIMARY_EXECUTION_PLAN_ID);
}

export function activateFallbackExecutionPlan(container) {
  const planState = getPlanState(container);
  if (!planState?.fallbackPlan) {
    return null;
  }
  return setActiveExecutionPlan(container, FINITENESS_FALLBACK_EXECUTION_PLAN_ID);
}

function resolveExecutionOverrides(options = {}) {
  return {
    disableCommandBatching: options.disableCommandBatching,
    disableMultiTokenDecode: options.disableMultiTokenDecode,
    batchSize: options.batchSize,
    stopCheckMode: options.stopCheckMode,
    maxTokens: options.maxTokens,
  };
}

export function resolveExecutionSessionPlan(container, options = {}) {
  const activePlan = resolveActiveExecutionPlan(container);
  const overrides = resolveExecutionOverrides(options);

  return {
    planId: activePlan.id,
    source: activePlan.source,
    kernelPath: activePlan.kernelPath,
    kernelPathId: activePlan.kernelPathId,
    activationDtype: activePlan.activationDtype,
    finitenessGuardEnabled: activePlan.finitenessGuardEnabled,
    finitenessAbsThreshold: activePlan.finitenessAbsThreshold,
    finitenessIncludeNonFinite: activePlan.finitenessIncludeNonFinite,
    deferredRoundingWindowTokens: activePlan.deferredRoundingWindowTokens,
    disableCommandBatching: overrides.disableCommandBatching ?? activePlan.defaultDisableCommandBatching,
    disableMultiTokenDecode: overrides.disableMultiTokenDecode ?? activePlan.defaultDisableMultiTokenDecode,
    batchSize: normalizePositiveInt(overrides.batchSize, activePlan.defaultBatchSize, 'batchSize'),
    stopCheckMode: normalizeStopCheckMode(overrides.stopCheckMode, activePlan.defaultStopCheckMode),
    maxTokens: normalizePositiveInt(overrides.maxTokens, activePlan.defaultMaxTokens, 'maxTokens'),
    readbackInterval: activePlan.readbackInterval,
    ringTokens: activePlan.ringTokens,
    ringStop: activePlan.ringStop,
    ringStaging: activePlan.ringStaging,
    overrides,
  };
}

export function rebaseExecutionSessionPlan(container, sessionPlan) {
  const overrides = sessionPlan?.overrides ?? {};
  return resolveExecutionSessionPlan(container, overrides);
}

export function isBatchDecodeEnabled(config) {
  return selectRuleValue('inference', 'execution', 'batchDecodeEnabled', {
    batchSize: config.batchSize,
    useGPU: config.useGPU,
    gpuSamplingAvailable: config.gpuSamplingAvailable,
    disableMultiTokenDecode: config.disableMultiTokenDecode,
    disableCommandBatching: config.disableCommandBatching,
    isBdpaPagedLayout: config.isBdpaPagedLayout === true,
    finitenessFallbackWindowOpen: config.finitenessFallbackWindowOpen === true,
  });
}

export function isDecodeRecorderEnabled(config) {
  return selectRuleValue('inference', 'execution', 'decodeRecorderEnabled', {
    hasDevice: config.hasDevice === true,
    debug: config.debug === true,
    disableCommandBatching: config.disableCommandBatching === true,
    kvLayout: config.kvLayout ?? null,
  });
}
