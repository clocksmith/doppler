import { selectRuleValue } from '../../../rules/rule-registry.js';
import { READBACK_MODES } from '../../../config/schema/execution-v1.schema.js';
import {
  resolveDeferredRoundingWindowTokens,
  resolveRangeAwareSelectiveWideningConfig,
} from './finiteness-policy.js';
import { log } from '../../../debug/index.js';

export const PRIMARY_EXECUTION_PLAN_ID = 'primary';
export const FINITENESS_FALLBACK_EXECUTION_PLAN_ID = 'finiteness_fallback';
const DEFAULT_MAX_TOKENS = 256;

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

function assertReadbackMode(value) {
  if (!value || !READBACK_MODES.includes(value)) {
    throw new Error(
      `[ExecutionPlan] readbackMode must be one of ${READBACK_MODES.join(', ')}; got ${JSON.stringify(value)}.`
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
    finitenessOnTrigger: finitenessPolicy.onTrigger,
    finitenessAbsThreshold: finitenessPolicy.absThreshold,
    finitenessIncludeNonFinite: finitenessPolicy.includeNonFinite,
    deferredRoundingWindowTokens,
    defaultDisableCommandBatching: batchingConfig.disableCommandBatching,
    defaultDisableMultiTokenDecode: generationConfig.disableMultiTokenDecode,
    defaultBatchSize: batchingConfig.batchSize,
    defaultStopCheckMode: batchingConfig.stopCheckMode,
    defaultMaxTokens: batchingConfig.maxTokens,
    readbackInterval: batchingConfig.readbackInterval,
    readbackMode: batchingConfig.readbackMode,
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

function validatePlanAgainstManifest(plan, manifest) {
  if (!manifest || !plan) return;
  const manifestInf = manifest.inference ?? manifest;
  const warnings = [];

  const manifestActivationDtype = manifestInf.session?.compute?.defaults?.activationDtype
    ?? manifestInf.compute?.activationDtype
    ?? manifestInf.quantizationInfo?.compute
    ?? null;
  if (manifestActivationDtype && plan.activationDtype
    && plan.activationDtype !== manifestActivationDtype) {
    warnings.push(
      `activationDtype: plan="${plan.activationDtype}" vs manifest="${manifestActivationDtype}"`
    );
  }

  if (warnings.length > 0) {
    log.warn(
      'ExecutionPlan',
      `Plan "${plan.id}" diverges from manifest: ${warnings.join('; ')}. ` +
      'Verify runtime config and manifest are in sync.'
    );
  }
}

export function compileExecutionPlanState(options) {
  const runtimeConfig = options?.runtimeConfig;
  const resolvedKernelPath = options?.resolvedKernelPath ?? null;
  const kernelPathSource = options?.kernelPathSource ?? 'none';
  const transformFallbackKernelPath = options?.fallbackKernelPath ?? null;
  const manifest = options?.manifest ?? null;

  if (!runtimeConfig?.inference) {
    throw new Error('[ExecutionPlan] runtimeConfig.inference is required.');
  }

  const inferenceConfig = runtimeConfig.inference;
  const computeConfig = inferenceConfig.compute;
  const generationConfig = inferenceConfig.generation ?? {};
  if (generationConfig.disableCommandBatching !== undefined) {
    throw new Error(
      '[ExecutionPlan] runtime.inference.generation.disableCommandBatching is removed. ' +
      'Use runtime.inference.session.decodeLoop.disableCommandBatching.'
    );
  }
  const sessionConfig = inferenceConfig.session ?? {};
  const decodeLoopConfig = sessionConfig.decodeLoop;
  if (!decodeLoopConfig || typeof decodeLoopConfig !== 'object') {
    throw new Error('[ExecutionPlan] runtimeConfig.inference.session.decodeLoop is required.');
  }

  const finitenessPolicy = resolveRangeAwareSelectiveWideningConfig(computeConfig);
  const deferredRoundingWindowTokens = resolveDeferredRoundingWindowTokens(computeConfig);
  const batchingConfig = {
    batchSize: decodeLoopConfig.batchSize,
    stopCheckMode: decodeLoopConfig.stopCheckMode,
    maxTokens: generationConfig.maxTokens ?? DEFAULT_MAX_TOKENS,
    readbackInterval: decodeLoopConfig.readbackInterval,
    readbackMode: assertReadbackMode(decodeLoopConfig.readbackMode),
    ringTokens: decodeLoopConfig.ringTokens,
    ringStop: decodeLoopConfig.ringStop,
    ringStaging: decodeLoopConfig.ringStaging,
    disableCommandBatching: assertOptionalBoolean(
      decodeLoopConfig.disableCommandBatching,
      'runtimeConfig.inference.session.decodeLoop.disableCommandBatching'
    ),
  };

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
  if (primaryPlan.finitenessGuardEnabled && primaryPlan.finitenessOnTrigger === 'fallback-plan') {
    const fallbackActivationDtype = resolveFallbackActivationDtype(primaryPlan.activationDtype);
    if (fallbackActivationDtype !== 'f32') {
      throw new Error(
        `[ExecutionPlan] finiteness fallback activation dtype must widen to "f32"; got "${fallbackActivationDtype}".`
      );
    }

    // Prefer transform-based fallback kernel path from execution-v1 compilation
    let fallbackKernelPathState;
    if (transformFallbackKernelPath) {
      fallbackKernelPathState = {
        kernelPath: transformFallbackKernelPath,
        kernelPathId: transformFallbackKernelPath.id ?? null,
        kernelPathSource: 'execution-v1-transform',
      };
      log.info(
        'ExecutionPlan',
        `Using transform-based finiteness fallback kernel path: ${transformFallbackKernelPath.id ?? 'inline'}`
      );
    } else {
      // Registry-based fallback was removed in Phase 3.
      // The finiteness fallback kernel path must come from execution-v1 transforms
      // or from an explicit finitenessFallbackKernelPathId on the kernel path object.
      throw new Error(
        `[ExecutionPlan] finiteness fallback kernel path required for "${primaryPlan.kernelPath?.id ?? 'unknown'}" ` +
        'but no transform-based fallback was provided. Use execution-v1 transforms or set ' +
        'finitenessFallbackKernelPathId explicitly on the kernel path object.'
      );
    }

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

  // Consistency check: validate the primary plan's kernel path source is recognized.
  const knownKernelPathSources = new Set([
    'none', 'config', 'model', 'manifest', 'execution-v1', 'execution-v1-transform',
  ]);
  if (primaryPlan.kernelPathSource && !knownKernelPathSources.has(primaryPlan.kernelPathSource)) {
    log.warn(
      'ExecutionPlan',
      `Primary plan kernelPathSource "${primaryPlan.kernelPathSource}" is not a recognized source. ` +
      `Known sources: ${[...knownKernelPathSources].join(', ')}.`
    );
  }
  // Validate that the plan's activation dtype is a recognized value.
  if (primaryPlan.activationDtype !== 'f16' && primaryPlan.activationDtype !== 'f32') {
    log.warn(
      'ExecutionPlan',
      `Primary plan activationDtype "${primaryPlan.activationDtype}" is not "f16" or "f32". ` +
      'Verify the manifest and runtime config produce a valid dtype.'
    );
  }

  // Validate plan against manifest when available (diagnostic only).
  if (manifest) {
    validatePlanAgainstManifest(primaryPlan, manifest);
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
    readbackInterval: assertOptionalPositiveInt(options.readbackInterval, 'readbackInterval'),
    ringTokens: assertOptionalPositiveInt(options.ringTokens, 'ringTokens'),
    ringStop: assertOptionalPositiveInt(options.ringStop, 'ringStop'),
    ringStaging: assertOptionalPositiveInt(options.ringStaging, 'ringStaging'),
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
    finitenessOnTrigger: activePlan.finitenessOnTrigger,
    finitenessAbsThreshold: activePlan.finitenessAbsThreshold,
    finitenessIncludeNonFinite: activePlan.finitenessIncludeNonFinite,
    deferredRoundingWindowTokens: activePlan.deferredRoundingWindowTokens,
    disableCommandBatching: overrides.disableCommandBatching ?? activePlan.defaultDisableCommandBatching,
    disableMultiTokenDecode: overrides.disableMultiTokenDecode ?? activePlan.defaultDisableMultiTokenDecode,
    batchSize: overrides.batchSize ?? activePlan.defaultBatchSize,
    stopCheckMode: overrides.stopCheckMode ?? activePlan.defaultStopCheckMode,
    maxTokens: overrides.maxTokens ?? activePlan.defaultMaxTokens,
    readbackInterval: overrides.readbackInterval ?? activePlan.readbackInterval,
    readbackMode: activePlan.readbackMode,
    ringTokens: overrides.ringTokens ?? activePlan.ringTokens,
    ringStop: overrides.ringStop ?? activePlan.ringStop,
    ringStaging: overrides.ringStaging ?? activePlan.ringStaging,
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
    hasLinearAttentionLayers: config.hasLinearAttentionLayers === true,
    selfSpeculationEnabled: config.selfSpeculationEnabled === true,
    hasRangeBackedPerLayerInputs: config.hasRangeBackedPerLayerInputs === true,
  });
}

export function resolveMaxBatchDecodeTokens(config) {
  const value = selectRuleValue('inference', 'execution', 'maxBatchDecodeTokens', {
    hasHotVocabularyBatchDecode: config.hasHotVocabularyBatchDecode === true,
    hasGpuSplitPerLayerInputs: config.hasGpuSplitPerLayerInputs === true,
    hasLinearAttentionLayers: config.hasLinearAttentionLayers === true,
  });
  if (value == null) {
    return null;
  }
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(
      `[ExecutionPlan] maxBatchDecodeTokens must be null or a positive integer; got ${JSON.stringify(value)}.`
    );
  }
  return value;
}

export function resolvePrefillRecorderChunkLayers(config) {
  const value = selectRuleValue('inference', 'execution', 'prefillRecorderChunkLayers', {
    hasGpuSplitPerLayerInputs: config.hasGpuSplitPerLayerInputs === true,
    numTokens: config.numTokens,
  });
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(
      `[ExecutionPlan] prefillRecorderChunkLayers must be a positive integer; got ${JSON.stringify(value)}.`
    );
  }
  return value;
}

export function isDecodeRecorderEnabled(config) {
  return selectRuleValue('inference', 'execution', 'decodeRecorderEnabled', {
    hasDevice: config.hasDevice === true,
    debug: config.debug === true,
    disableCommandBatching: config.disableCommandBatching === true,
    kvLayout: config.kvLayout ?? null,
  });
}

export function isProfileDecodeRecorderEnabled(config) {
  return selectRuleValue('inference', 'execution', 'profileDecodeRecorderEnabled', {
    hasDevice: config.hasDevice === true,
    debug: config.debug === true,
    disableCommandBatching: config.disableCommandBatching === true,
    kvLayout: config.kvLayout ?? null,
  });
}
