import { resolveKernelPath } from '../../../config/kernel-path-loader.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';
import {
  resolveDeferredRoundingWindowTokens,
  resolveRangeAwareSelectiveWideningConfig,
} from './finiteness-policy.js';

export const PRIMARY_EXECUTION_PLAN_ID = 'primary';
export const FINITENESS_FALLBACK_EXECUTION_PLAN_ID = 'finiteness_fallback';

function assertOptionalBoolean(value, label) {
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== 'boolean') {
    throw new Error(`[ExecutionPlan] ${label} must be boolean when provided; got ${JSON.stringify(value)}.`);
  }
  return value;
}

function assertOptionalPositiveInt(value, label) {
  if (value === undefined) {
    return undefined;
  }
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`[ExecutionPlan] ${label} must be a positive integer when provided; got ${JSON.stringify(value)}.`);
  }
  return value;
}

function assertOptionalStopCheckMode(value) {
  if (value === undefined) {
    return undefined;
  }
  if (value !== 'batch' && value !== 'per-token') {
    throw new Error(
      `[ExecutionPlan] stopCheckMode must be "batch" or "per-token" when provided; got ${JSON.stringify(value)}.`
    );
  }
  return value;
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
    throw new Error(
      '[ExecutionPlan] F16 finiteness fallback requires a primary kernel path with a stable id. ' +
      'Add a registered kernelPath id and a finiteness fallback rule.'
    );
  }

  const explicitFallbackKernelPathId = typeof primaryKernelPath?.finitenessFallbackKernelPathId === 'string'
    && primaryKernelPath.finitenessFallbackKernelPathId.length > 0
    ? primaryKernelPath.finitenessFallbackKernelPathId
    : null;

  const fallbackKernelPathId = explicitFallbackKernelPathId ?? selectRuleValue(
    'inference',
    'kernelPath',
    'finitenessFallback',
    { kernelPathId: primaryKernelPathId }
  );

  if (typeof fallbackKernelPathId !== 'string' || fallbackKernelPathId.length === 0) {
    throw new Error(
      `[ExecutionPlan] Missing finiteness fallback kernel path mapping for "${primaryKernelPathId}". ` +
      'Add an explicit rule in src/rules/inference/kernel-path.rules.json.'
    );
  }

  if (fallbackKernelPathId === primaryKernelPathId) {
    throw new Error(
      `[ExecutionPlan] Invalid finiteness fallback mapping for "${primaryKernelPathId}": ` +
      `fallback kernel path resolves to itself. Add an explicit widening path.`
    );
  }

  try {
    const kernelPath = resolveKernelPath(fallbackKernelPathId);
    return {
      kernelPath,
      kernelPathId: fallbackKernelPathId,
      kernelPathSource: 'rule',
    };
  } catch (error) {
    throw new Error(
      `[ExecutionPlan] Failed to resolve finiteness fallback kernel path "${fallbackKernelPathId}" ` +
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
    if (fallbackActivationDtype !== 'f32') {
      throw new Error(
        `[ExecutionPlan] finiteness fallback activation dtype must widen to "f32"; got "${fallbackActivationDtype}".`
      );
    }
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

    if (fallbackPlan.finitenessGuardEnabled) {
      throw new Error('[ExecutionPlan] finiteness fallback plan cannot enable finiteness guard.');
    }
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
    disableCommandBatching: assertOptionalBoolean(
      options.disableCommandBatching,
      'disableCommandBatching'
    ),
    disableMultiTokenDecode: assertOptionalBoolean(
      options.disableMultiTokenDecode,
      'disableMultiTokenDecode'
    ),
    batchSize: assertOptionalPositiveInt(options.batchSize, 'batchSize'),
    stopCheckMode: assertOptionalStopCheckMode(options.stopCheckMode),
    maxTokens: assertOptionalPositiveInt(options.maxTokens, 'maxTokens'),
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
    batchSize: overrides.batchSize ?? activePlan.defaultBatchSize,
    stopCheckMode: overrides.stopCheckMode ?? activePlan.defaultStopCheckMode,
    maxTokens: overrides.maxTokens ?? activePlan.defaultMaxTokens,
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
