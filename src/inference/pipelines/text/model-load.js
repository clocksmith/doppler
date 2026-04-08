import { markWarmed as markKernelCacheWarmed } from '../../../gpu/kernel-selection-cache.js';
import { getKernelCapabilities } from '../../../gpu/device.js';
import { log } from '../../../debug/index.js';
import {
  resolveKernelPath,
  getKernelPathStats,
  getKernelPathActivationDtype,
  getKernelPathOutputDtype,
  getKernelPathKVDtype,
  setActiveKernelPath,
} from '../../../config/kernel-path-loader.js';
import { autoTuneKernels, prewarmKernels } from '../../../gpu/kernels/index.js';
import { KERNEL_CONFIGS } from '../../../gpu/kernels/kernel-configs.js';
import { initTokenizer } from './init.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';
import { mergeRuntimeValues } from '../../../config/runtime-merge.js';
import { READBACK_MODES } from '../../../config/schema/execution-v1.schema.js';

function validateKernelWarmupMode(mode) {
  if (mode !== 'parallel' && mode !== 'sequential') {
    throw new Error(
      `runtime.shared.kernelWarmup.prewarmMode must be "parallel" or "sequential"; got "${mode}".`
    );
  }
}

function normalizePositiveInt(value) {
  if (!Number.isFinite(value)) return null;
  const normalized = Math.floor(value);
  return normalized >= 1 ? normalized : null;
}

function normalizeStopCheckMode(value) {
  if (value === 'batch' || value === 'per-token') return value;
  return null;
}

function normalizeReadbackInterval(value) {
  if (value == null) return null;
  return normalizePositiveInt(value);
}

function normalizeBoolean(value) {
  return typeof value === 'boolean' ? value : null;
}

function parseSessionDecodeLoopOptionalPositiveInt(value, label, modelId) {
  if (value === undefined) {
    return undefined;
  }
  if (value === null) {
    return null;
  }
  const normalized = normalizePositiveInt(value);
  if (normalized == null) {
    throw new Error(
      `Manifest "${modelId}" inference.session.decodeLoop.${label} must be a positive integer or null.`
    );
  }
  return normalized;
}

function parseSessionDecodeLoopOptionalBoolean(value, label, modelId) {
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== 'boolean') {
    throw new Error(
      `Manifest "${modelId}" inference.session.decodeLoop.${label} must be a boolean when provided.`
    );
  }
  return value;
}

function requireSessionDecodeLoopPositiveInt(value, label, modelId) {
  const normalized = normalizePositiveInt(value);
  if (normalized == null) {
    throw new Error(`Manifest "${modelId}" inference.session.decodeLoop.${label} must be a positive integer.`);
  }
  return normalized;
}

function requireSessionDecodeLoopStopCheckMode(value, modelId) {
  const normalized = normalizeStopCheckMode(value);
  if (normalized == null) {
    throw new Error(
      `Manifest "${modelId}" inference.session.decodeLoop.stopCheckMode must be "batch" or "per-token".`
    );
  }
  return normalized;
}

function buildResolvedDecodeLoopRuntimePatch(runtimeConfig, manifest) {
  const resolvedSession = mergeRuntimeValues(
    manifest?.inference?.session ?? {},
    runtimeConfig?.inference?.session ?? {}
  );
  const decodeLoop = resolvedSession?.decodeLoop;
  if (decodeLoop == null) {
    return null;
  }
  const modelId = String(manifest?.modelId ?? 'unknown').trim() || 'unknown';
  if (typeof decodeLoop !== 'object') {
    throw new Error(
      `Manifest "${modelId}" inference.session.decodeLoop must be an object when provided.`
    );
  }
  const batchSize = requireSessionDecodeLoopPositiveInt(decodeLoop.batchSize, 'batchSize', modelId);
  const stopCheckMode = requireSessionDecodeLoopStopCheckMode(decodeLoop.stopCheckMode, modelId);
  const readbackInterval = requireSessionDecodeLoopPositiveInt(
    decodeLoop.readbackInterval,
    'readbackInterval',
    modelId
  );
  const disableCommandBatching = parseSessionDecodeLoopOptionalBoolean(
    decodeLoop.disableCommandBatching,
    'disableCommandBatching',
    modelId
  );

  const readbackMode = decodeLoop.readbackMode;
  if (!readbackMode || !READBACK_MODES.includes(readbackMode)) {
    throw new Error(
      `DopplerConfigError: Manifest "${modelId}" inference.session.decodeLoop.readbackMode ` +
      `is required and must be one of ${READBACK_MODES.join(', ')}; got ${JSON.stringify(readbackMode)}. ` +
      'Set it explicitly in the manifest session.decodeLoop.'
    );
  }
  const submitLatencyThresholdMs = decodeLoop.submitLatencyThresholdMs ?? null;

  const batchingPatch = {
    batchSize,
    stopCheckMode,
    readbackInterval,
  };
  const ringTokens = parseSessionDecodeLoopOptionalPositiveInt(
    decodeLoop.ringTokens,
    'ringTokens',
    modelId
  );
  if (ringTokens !== undefined) {
    batchingPatch.ringTokens = ringTokens;
  }
  const ringStop = parseSessionDecodeLoopOptionalPositiveInt(
    decodeLoop.ringStop,
    'ringStop',
    modelId
  );
  if (ringStop !== undefined) {
    batchingPatch.ringStop = ringStop;
  }
  const ringStaging = parseSessionDecodeLoopOptionalPositiveInt(
    decodeLoop.ringStaging,
    'ringStaging',
    modelId
  );
  if (ringStaging !== undefined) {
    batchingPatch.ringStaging = ringStaging;
  }

  return {
    session: {
      ...resolvedSession,
      decodeLoop: {
        ...decodeLoop,
        batchSize,
        stopCheckMode,
        readbackInterval,
        readbackMode,
        submitLatencyThresholdMs,
        ...(ringTokens !== undefined ? { ringTokens } : {}),
        ...(ringStop !== undefined ? { ringStop } : {}),
        ...(ringStaging !== undefined ? { ringStaging } : {}),
        ...(disableCommandBatching !== undefined ? { disableCommandBatching } : {}),
      },
    },
    batching: batchingPatch,
  };
}

export function applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, modelConfig) {
  void modelConfig;
  const patch = buildResolvedDecodeLoopRuntimePatch(runtimeConfig, manifest);
  if (!patch) {
    return runtimeConfig;
  }

  // Resolve readbackMode. "auto" runs the submit probe and resolves to a
  // concrete mode. Validation runs once here at pipeline init.
  const dl = patch.session.decodeLoop;
  let resolvedReadbackMode = dl.readbackMode;

  if (resolvedReadbackMode === 'overlapped') {
    const rs = dl.ringStaging ?? 0;
    if (rs < 2) {
      throw new Error(
        `DopplerConfigError: readbackMode "overlapped" requires ringStaging >= 2, got ${rs}.`
      );
    }
  }

  if (resolvedReadbackMode === 'auto') {
    const thresholdMs = dl.submitLatencyThresholdMs;
    if (thresholdMs == null) {
      throw new Error(
        'DopplerConfigError: readbackMode "auto" requires submitLatencyThresholdMs to be set.'
      );
    }
    let probeMs = null;
    try {
      probeMs = getKernelCapabilities().submitProbeMs;
    } catch {
      // Device not initialized yet — fall back to sequential.
    }
    if (probeMs != null && probeMs > thresholdMs) {
      const rs = dl.ringStaging ?? 0;
      if (rs < 2) {
        log.info(
          'Pipeline',
          `readbackMode auto: probe ${probeMs.toFixed(1)}ms > threshold ${thresholdMs}ms ` +
          `but ringStaging ${rs} < 2; staying sequential`
        );
        resolvedReadbackMode = 'sequential';
      } else {
        resolvedReadbackMode = 'overlapped';
      }
    } else {
      resolvedReadbackMode = 'sequential';
    }
    log.info(
      'Pipeline',
      `readbackMode resolved to ${resolvedReadbackMode} ` +
      `(probe: ${probeMs != null ? probeMs.toFixed(1) + 'ms' : 'unavailable'}, threshold: ${thresholdMs}ms)`
    );
  }

  dl.readbackMode = resolvedReadbackMode;
  patch.batching.readbackMode = resolvedReadbackMode;

  const nextRuntimeConfig = mergeRuntimeValues(runtimeConfig, {
    inference: {
      session: patch.session,
      batching: patch.batching,
    },
  });
  log.info(
    'Pipeline',
    `Resolved session applied (${manifest?.modelId ?? 'unknown'}): ` +
    `batchSize=${patch.batching.batchSize}, stopCheckMode=${patch.batching.stopCheckMode}, ` +
    `readbackInterval=${patch.batching.readbackInterval}, ` +
    `disableCommandBatching=${patch.session.decodeLoop?.disableCommandBatching === true}`
  );
  return nextRuntimeConfig;
}

export async function runKernelWarmup(options) {
  const { useGPU, kernelWarmup, modelConfig } = options;
  if (!useGPU || !kernelWarmup) {
    return;
  }
  if (kernelWarmup.prewarm) {
    const mode = kernelWarmup.prewarmMode;
    validateKernelWarmupMode(mode);
    log.info('Pipeline', `Kernel prewarm enabled (mode=${mode})`);
    try {
      await prewarmKernels({ mode });
      markKernelCacheWarmed();
    } catch (e) {
      log.warn('Pipeline', `Kernel prewarm failed: ${ (e).message}`);
    }
  }
  if (kernelWarmup.autoTune) {
    log.info('Pipeline', 'Kernel auto-tune enabled');
    try {
      await autoTuneKernels(modelConfig);
      markKernelCacheWarmed();
    } catch (e) {
      log.warn('Pipeline', `Kernel auto-tune failed: ${ (e).message}`);
    }
  }
}

function normalizeKernelPathSourceHint(value) {
  const normalized = String(value ?? '').trim().toLowerCase();
  if (normalized === 'runtime') return 'config';
  return normalized || 'none';
}

function resolveKernelPathSource(runtimeConfigKernelPath, runtimeKernelPathSourceHint, modelKernelPath) {
  if (runtimeConfigKernelPath) {
    const sourceHint = normalizeKernelPathSourceHint(runtimeKernelPathSourceHint);
    if (sourceHint !== 'none') return sourceHint;
    return 'config';
  }
  if (modelKernelPath) return 'model';
  return 'manifest';
}

function normalizeKernelFileName(kernel) {
  const normalized = String(kernel ?? '').trim();
  if (!normalized) return '';
  const parts = normalized.split('/');
  return parts[parts.length - 1] ?? normalized;
}

function buildKernelRequiredFeaturesByShaderEntry() {
  const index = new Map();
  for (const variantsByOperation of Object.values(KERNEL_CONFIGS ?? {})) {
    if (!variantsByOperation || typeof variantsByOperation !== 'object') continue;
    for (const variantConfig of Object.values(variantsByOperation)) {
      if (!variantConfig || typeof variantConfig !== 'object') continue;
      const shaderFile = normalizeKernelFileName(variantConfig.shaderFile);
      if (!shaderFile) continue;
      const entryPoint = String(variantConfig.entryPoint ?? 'main').trim() || 'main';
      const key = `${shaderFile}#${entryPoint}`;
      const requires = index.get(key) ?? new Set();
      for (const requirement of variantConfig.requires ?? []) {
        const normalizedRequirement = String(requirement ?? '').trim();
        if (!normalizedRequirement) continue;
        requires.add(normalizedRequirement);
      }
      index.set(key, requires);
    }
  }
  return index;
}

const KERNEL_REQUIRED_FEATURES_BY_SHADER_ENTRY = buildKernelRequiredFeaturesByShaderEntry();

function collectKernelPathSteps(kernelPath) {
  const steps = [];
  const append = (list) => {
    for (const step of list ?? []) {
      if (!step || typeof step !== 'object') continue;
      steps.push(step);
    }
  };
  append(kernelPath?.decode?.steps);
  append(kernelPath?.prefill?.steps);
  append(kernelPath?.preLayer);
  append(kernelPath?.postLayer);
  append(kernelPath?.sampling);
  for (const override of kernelPath?.layerOverrides ?? []) {
    append(override?.steps);
    append(override?.decode?.steps);
    append(override?.prefill?.steps);
  }
  return steps;
}

function findKernelPathUnsupportedFeatureUsages(kernelPath, capabilities) {
  const offenders = [];
  const seen = new Set();
  const hasSubgroups = capabilities?.hasSubgroups === true;
  const hasF16 = capabilities?.hasF16 === true;
  for (const step of collectKernelPathSteps(kernelPath)) {
    const kernelFile = normalizeKernelFileName(step.kernel);
    if (!kernelFile) continue;
    const entryPoint = String(step.entry ?? 'main').trim() || 'main';
    const key = `${kernelFile}#${entryPoint}`;
    const requirements = KERNEL_REQUIRED_FEATURES_BY_SHADER_ENTRY.get(key);
    if (!requirements) continue;

    for (const requirement of requirements) {
      let supported = true;
      if (requirement === 'subgroups') {
        supported = hasSubgroups;
      } else if (requirement === 'shader-f16') {
        supported = hasF16;
      } else {
        continue;
      }
      if (supported) continue;

      const dedupeKey = `${key}:${requirement}:${step.op ?? 'unknown'}`;
      if (seen.has(dedupeKey)) continue;
      seen.add(dedupeKey);
      offenders.push({
        op: String(step.op ?? 'unknown'),
        kernel: kernelFile,
        entry: entryPoint,
        requirement,
      });
    }
  }
  return offenders;
}

function summarizeUnsupportedKernelUsages(usages) {
  const summarized = usages
    .slice(0, 8)
    .map((usage) => `${usage.op}:${usage.kernel}#${usage.entry} (${usage.requirement})`)
    .join(', ');
  const remaining = usages.length > 8 ? ` (+${usages.length - 8} more)` : '';
  return `${summarized}${remaining}`;
}

function assertKernelPathFeatureCompatibility(
  configuredKernelPathRef,
  effectiveKernelPathRef,
  resolvedKernelPath,
  kernelPathSource,
  capabilities,
  kernelPathPolicy
) {
  const unsupportedUsages = findKernelPathUnsupportedFeatureUsages(resolvedKernelPath, capabilities);
  if (unsupportedUsages.length === 0) return;

  const sourceScope = kernelPathPolicy.sourceScope ?? kernelPathPolicy.allowSources ?? [];
  const policyAllowsSource = kernelPathPolicy.mode === 'capability-aware'
    && sourceScope.includes(kernelPathSource);
  const remapRequested = policyAllowsSource && kernelPathPolicy.onIncompatible === 'remap';
  const remapApplied = typeof configuredKernelPathRef === 'string'
    && typeof effectiveKernelPathRef === 'string'
    && configuredKernelPathRef !== effectiveKernelPathRef;
  const summary = summarizeUnsupportedKernelUsages(unsupportedUsages);

  if (remapRequested && !remapApplied) {
    throw new Error(
      `KernelPath "${resolvedKernelPath?.id ?? 'unknown'}" requires unsupported GPU features (${summary}) ` +
      `and no explicit auto-select remap matched for source "${kernelPathSource}". ` +
      'Add a kernel-path remap rule or choose a compatible kernelPath.'
    );
  }

  throw new Error(
    `KernelPath "${resolvedKernelPath?.id ?? 'unknown'}" requires unsupported GPU features: ${summary}. ` +
    'Choose a compatible kernelPath or enable explicit capability remap rules.'
  );
}

function normalizeKernelDtype(value) {
  if (!value) return null;
  const lower = String(value).trim().toLowerCase();
  if (!lower) return null;
  return selectRuleValue('inference', 'dtype', 'f16OrF32FromDtypeAlias', {
    dtype: lower,
    fallback: null,
  });
}

function buildKernelPathDtypeContract(resolvedKernelPath) {
  if (!resolvedKernelPath) {
    return null;
  }
  const activationDtype = normalizeKernelDtype(getKernelPathActivationDtype(resolvedKernelPath));
  const outputDtype = normalizeKernelDtype(
    getKernelPathOutputDtype(resolvedKernelPath) ?? activationDtype
  );
  const kvDtype = normalizeKernelDtype(getKernelPathKVDtype(resolvedKernelPath) ?? activationDtype);
  if (!activationDtype && !outputDtype && !kvDtype) {
    return null;
  }
  return {
    activationDtype,
    outputDtype,
    kvDtype,
  };
}

function describeKernelPathDtypeMismatch(contract, current) {
  const mismatches = [];
  if (contract.activationDtype && current.activationDtype !== contract.activationDtype) {
    mismatches.push(
      `runtime.inference.compute.activationDtype=${current.activationDtype ?? 'unset'} ` +
      `(expected ${contract.activationDtype})`
    );
  }
  if (contract.kvDtype && current.kvDtype !== contract.kvDtype) {
    mismatches.push(
      `runtime.inference.session.kvcache.kvDtype=${current.kvDtype ?? 'unset'} ` +
      `(expected ${contract.kvDtype})`
    );
  }
  if (contract.outputDtype && current.outputDtype !== contract.outputDtype) {
    mismatches.push(
      `runtime.inference.session.compute.defaults.outputDtype=${current.outputDtype ?? 'unset'} ` +
      `(expected ${contract.outputDtype})`
    );
  }
  return mismatches;
}

function describeExplicitRuntimeDtypeMismatch(contract, explicitRuntime) {
  const mismatches = [];
  if (
    contract.activationDtype
    && explicitRuntime.activationDtype != null
    && explicitRuntime.activationDtype !== contract.activationDtype
  ) {
    mismatches.push(
      `runtime.inference.session.compute.defaults.activationDtype=${explicitRuntime.activationDtype} ` +
      `(expected ${contract.activationDtype})`
    );
  }
  if (
    contract.kvDtype
    && explicitRuntime.kvDtype != null
    && explicitRuntime.kvDtype !== contract.kvDtype
  ) {
    mismatches.push(
      `runtime.inference.session.kvcache.kvDtype=${explicitRuntime.kvDtype} ` +
      `(expected ${contract.kvDtype})`
    );
  }
  if (
    contract.outputDtype
    && explicitRuntime.outputDtype != null
    && explicitRuntime.outputDtype !== contract.outputDtype
  ) {
    mismatches.push(
      `runtime.inference.session.compute.defaults.outputDtype=${explicitRuntime.outputDtype} ` +
      `(expected ${contract.outputDtype})`
    );
  }
  return mismatches;
}

function assertManifestKernelPathDtypeCompatibility(manifest, resolvedKernelPath, kernelPathSource) {
  if (!resolvedKernelPath) return;
  if (kernelPathSource === 'config') return;
  if (kernelPathSource !== 'model' && kernelPathSource !== 'manifest') return;

  const manifestCompute = normalizeKernelDtype(manifest?.quantizationInfo?.compute);
  const kernelActivation = normalizeKernelDtype(getKernelPathActivationDtype(resolvedKernelPath));
  if (!manifestCompute || !kernelActivation) return;
  if (manifestCompute === kernelActivation) return;

  throw new Error(
    `Manifest kernel path dtype mismatch for "${manifest?.modelId ?? 'unknown'}": ` +
    `quantizationInfo.compute=${manifestCompute} but ` +
    `kernelPath="${resolvedKernelPath.id}" uses activationDtype=${kernelActivation}. ` +
    'Re-convert the model or set runtime.inference.kernelPath explicitly.'
  );
}

function getKernelCapabilitiesSafe() {
  try {
    return getKernelCapabilities();
  } catch {
    return null;
  }
}

const DEFAULT_KERNEL_PATH_POLICY = Object.freeze({
  mode: 'locked',
  sourceScope: Object.freeze(['model', 'manifest']),
  onIncompatible: 'error',
});

function normalizeKernelPathPolicyMode(value) {
  const normalized = String(value ?? '').trim().toLowerCase();
  if (normalized === 'capability-aware') {
    return 'capability-aware';
  }
  return 'locked';
}

function normalizeKernelPathPolicySourceScope(value) {
  if (!Array.isArray(value)) {
    return [...DEFAULT_KERNEL_PATH_POLICY.sourceScope];
  }
  const normalized = new Set();
  for (const source of value) {
    const normalizedSource = String(source ?? '').trim().toLowerCase();
    if (normalizedSource === 'runtime') {
      normalized.add('config');
    } else if (normalizedSource && normalizedSource !== 'none') {
      normalized.add(normalizedSource);
    }
  }
  if (normalized.size === 0) {
    return [...DEFAULT_KERNEL_PATH_POLICY.sourceScope];
  }
  return [...normalized];
}

function normalizeKernelPathPolicyOnIncompatible(value) {
  const normalized = String(value ?? '').trim().toLowerCase();
  if (normalized === 'remap') {
    return 'remap';
  }
  return 'error';
}

function resolveKernelPathPolicy(policy) {
  if (!policy || typeof policy !== 'object' || Array.isArray(policy)) {
    return {
      mode: DEFAULT_KERNEL_PATH_POLICY.mode,
      sourceScope: [...DEFAULT_KERNEL_PATH_POLICY.sourceScope],
      allowSources: [...DEFAULT_KERNEL_PATH_POLICY.sourceScope],
      onIncompatible: DEFAULT_KERNEL_PATH_POLICY.onIncompatible,
    };
  }

  const sourceScope = normalizeKernelPathPolicySourceScope(
    policy.sourceScope ?? policy.allowSources
  );

  return {
    mode: normalizeKernelPathPolicyMode(policy.mode),
    sourceScope,
    allowSources: [...sourceScope],
    onIncompatible: normalizeKernelPathPolicyOnIncompatible(policy.onIncompatible),
  };
}

function getExplicitRuntimeDtypeOverrides(runtimeOverrides) {
  return {
    activationDtype: normalizeKernelDtype(
      runtimeOverrides?.inference?.session?.compute?.defaults?.activationDtype
      ?? runtimeOverrides?.inference?.compute?.activationDtype
    ),
    kvDtype: normalizeKernelDtype(
      runtimeOverrides?.inference?.session?.kvcache?.kvDtype
    ),
    outputDtype: normalizeKernelDtype(
      runtimeOverrides?.inference?.session?.compute?.defaults?.outputDtype
    ),
  };
}

function applyKernelPathRuntimeDtypeContract(
  resolvedKernelPath,
  runtimeConfig,
  runtimeOverrides,
  kernelPathSource,
  modelId
) {
  const contract = buildKernelPathDtypeContract(resolvedKernelPath);
  if (!contract) {
    return runtimeConfig;
  }

  const current = {
    activationDtype: normalizeKernelDtype(runtimeConfig.inference?.compute?.activationDtype),
    kvDtype: normalizeKernelDtype(runtimeConfig.inference?.session?.kvcache?.kvDtype),
    outputDtype: normalizeKernelDtype(runtimeConfig.inference?.session?.compute?.defaults?.outputDtype),
  };
  const mismatches = describeKernelPathDtypeMismatch(contract, current);
  if (mismatches.length === 0) {
    return runtimeConfig;
  }

  const explicitRuntime = getExplicitRuntimeDtypeOverrides(runtimeOverrides);
  const explicitMismatches = describeExplicitRuntimeDtypeMismatch(contract, explicitRuntime);

  if (kernelPathSource === 'config') {
    throw new Error(
      `KernelPath "${resolvedKernelPath?.id ?? 'unknown'}" selected from ${kernelPathSource} ` +
      `requires explicit matching runtime dtypes for "${modelId}". ` +
      `Mismatches: ${mismatches.join('; ')}. ` +
      'Set runtime.inference.session.compute.defaults.activationDtype, runtime.inference.session.kvcache.kvDtype, ' +
      'and runtime.inference.session.compute.defaults.outputDtype to match the kernel path.'
    );
  }

  if (explicitMismatches.length > 0) {
    throw new Error(
      `Manifest/model kernelPath "${resolvedKernelPath?.id ?? 'unknown'}" for "${modelId}" ` +
      `conflicts with runtime dtype overrides. Mismatches: ${explicitMismatches.join('; ')}. ` +
      'Either remove the runtime dtype override or set it to match the kernel path.'
    );
  }

  throw new Error(
    `Manifest/model kernelPath "${resolvedKernelPath?.id ?? 'unknown'}" for "${modelId}" ` +
    `requires matching runtime dtypes. Mismatches: ${mismatches.join('; ')}. ` +
    'Set runtime.inference.session.compute.defaults.activationDtype, ' +
    'runtime.inference.session.kvcache.kvDtype, and ' +
    'runtime.inference.session.compute.defaults.outputDtype to match the kernel path. ' +
    'Runtime dtype auto-rewrites are not allowed.'
  );
}

export function resolveKernelPathState(options) {
  const {
    manifest,
    runtimeConfig,
    runtimeOverrides = null,
    modelConfig,
    kernelCapabilities = null,
  } = options;

  log.debug(
    'Pipeline',
    `kernelPath sources: config=${runtimeConfig.inference.kernelPath}, model=${modelConfig.kernelPath}`
  );

  // In normal operation with execution-v1 manifests, modelConfig.kernelPath is null
  // and capability adaptation is handled by execution graph transforms.
  // The runtime override path (runtime.inference.kernelPath) is preserved for
  // explicit user overrides with inline kernel path objects.
  const configuredKernelPathRef = runtimeConfig.inference.kernelPath
    ?? modelConfig.kernelPath
    ?? null;
  let kernelPathSource = 'none';
  let resolvedKernelPath = null;
  const kernelPathPolicy = resolveKernelPathPolicy(runtimeConfig?.inference?.kernelPathPolicy);

  if (configuredKernelPathRef) {
    kernelPathSource = resolveKernelPathSource(
      runtimeConfig.inference.kernelPath,
      runtimeConfig.inference.kernelPathSource,
      modelConfig.kernelPath
    );

    // Registry-based auto-select removed (Phase 3). The configured ref is used directly.
    // Capability adaptation for execution-v1 manifests is handled by execution graph transforms.
    try {
      resolvedKernelPath = resolveKernelPath(configuredKernelPathRef);
    } catch (e) {
      throw new Error(`KernelPath resolution failed for '${configuredKernelPathRef}': ${ (e).message}`);
    }

    const capabilities = kernelCapabilities && typeof kernelCapabilities === 'object'
      ? kernelCapabilities
      : getKernelCapabilitiesSafe();

    if (capabilities) {
      assertKernelPathFeatureCompatibility(
        configuredKernelPathRef,
        configuredKernelPathRef,
        resolvedKernelPath,
        kernelPathSource,
        capabilities,
        kernelPathPolicy
      );
    }

    const stats = getKernelPathStats(resolvedKernelPath);
    log.info(
      'Pipeline',
      `KernelPath: ${resolvedKernelPath.id} (${stats.decodeSteps} decode steps, ${stats.uniqueKernels} kernels, source=${kernelPathSource})`
    );
    assertManifestKernelPathDtypeCompatibility(manifest, resolvedKernelPath, kernelPathSource);
  } else {
    log.info('Pipeline', 'KernelPath: none (execution graph transforms handle capability adaptation)');
  }

  const nextRuntimeConfig = applyKernelPathRuntimeDtypeContract(
    resolvedKernelPath,
    runtimeConfig,
    runtimeOverrides,
    kernelPathSource,
    String(manifest?.modelId ?? 'unknown').trim() || 'unknown'
  );
  return {
    resolvedKernelPath,
    kernelPathSource,
    kernelPathPolicy,
    runtimeConfig: nextRuntimeConfig,
  };
}

export function activateKernelPathState(kernelPathState) {
  setActiveKernelPath(
    kernelPathState?.resolvedKernelPath ?? null,
    kernelPathState?.kernelPathSource ?? 'none',
    kernelPathState?.kernelPathPolicy ?? null
  );
}

export function resolveAndActivateKernelPath(options) {
  const state = resolveKernelPathState(options);
  activateKernelPathState(state);
  return state;
}

export async function initTokenizerFromManifest(manifest, baseUrl, storageContext = null) {
  return initTokenizer(manifest, {
    baseUrl: baseUrl ?? undefined,
    tokenizerHints: null,
    storageContext: storageContext ?? undefined,
  });
}
