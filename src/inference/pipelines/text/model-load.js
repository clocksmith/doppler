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
import { resolveCapabilityKernelPathRef, resolveKernelPathPolicy } from './kernel-path-auto-select.js';
import { initTokenizer } from './init.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';
import { mergeRuntimeValues } from '../../../config/runtime-merge.js';
import {
  DEFAULT_BATCHING_DEFAULTS,
  DEFAULT_COMPUTE_DEFAULTS,
  DEFAULT_GENERATION_CONFIG,
} from '../../../config/schema/inference-defaults.schema.js';
import { DEFAULT_KVCACHE_CONFIG } from '../../../config/schema/kvcache.schema.js';
import { DEFAULT_EXECUTION_V1_COMPUTE_DEFAULTS } from '../../../config/schema/execution-v1.schema.js';

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

function parseManifestDecodeLoopOptionalPositiveInt(value, label, modelId) {
  if (value === undefined) {
    return undefined;
  }
  if (value === null) {
    return null;
  }
  const normalized = normalizePositiveInt(value);
  if (normalized == null) {
    throw new Error(
      `Manifest "${modelId}" inference.sessionDefaults.decodeLoop.${label} must be a positive integer or null.`
    );
  }
  return normalized;
}

function parseManifestDecodeLoopOptionalBoolean(value, label, modelId) {
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== 'boolean') {
    throw new Error(
      `Manifest "${modelId}" inference.sessionDefaults.decodeLoop.${label} must be a boolean when provided.`
    );
  }
  return value;
}

function requireGlobalBatchingDefault(value, label) {
  const normalized = normalizePositiveInt(value);
  if (normalized == null) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return normalized;
}

function requireGlobalStopCheckMode(value, label) {
  const normalized = normalizeStopCheckMode(value);
  if (normalized == null) {
    throw new Error(`${label} must be "batch" or "per-token".`);
  }
  return normalized;
}

const GLOBAL_DEFAULT_BATCHING = Object.freeze({
  batchSize: requireGlobalBatchingDefault(
    DEFAULT_BATCHING_DEFAULTS.batchSize,
    'DEFAULT_BATCHING_DEFAULTS.batchSize'
  ),
  stopCheckMode: requireGlobalStopCheckMode(
    DEFAULT_BATCHING_DEFAULTS.stopCheckMode,
    'DEFAULT_BATCHING_DEFAULTS.stopCheckMode'
  ),
  readbackInterval: requireGlobalBatchingDefault(
    DEFAULT_BATCHING_DEFAULTS.readbackInterval,
    'DEFAULT_BATCHING_DEFAULTS.readbackInterval'
  ),
  ringTokens: requireGlobalBatchingDefault(
    DEFAULT_BATCHING_DEFAULTS.ringTokens,
    'DEFAULT_BATCHING_DEFAULTS.ringTokens'
  ),
  ringStop: requireGlobalBatchingDefault(
    DEFAULT_BATCHING_DEFAULTS.ringStop,
    'DEFAULT_BATCHING_DEFAULTS.ringStop'
  ),
  ringStaging: requireGlobalBatchingDefault(
    DEFAULT_BATCHING_DEFAULTS.ringStaging,
    'DEFAULT_BATCHING_DEFAULTS.ringStaging'
  ),
});

const GLOBAL_DEFAULT_GENERATION = Object.freeze({
  disableCommandBatching: DEFAULT_GENERATION_CONFIG.disableCommandBatching === true,
});

const GLOBAL_DEFAULT_KERNEL_PATH_DTYPES = Object.freeze({
  activationDtype: DEFAULT_COMPUTE_DEFAULTS.activationDtype,
  kvDtype: DEFAULT_KVCACHE_CONFIG.kvDtype,
  outputDtype: DEFAULT_EXECUTION_V1_COMPUTE_DEFAULTS.outputDtype,
});

function isRuntimeBatchingAtGlobalDefaults(batching) {
  if (!batching || typeof batching !== 'object') {
    return false;
  }
  return normalizePositiveInt(batching.batchSize) === GLOBAL_DEFAULT_BATCHING.batchSize
    && normalizeStopCheckMode(batching.stopCheckMode) === GLOBAL_DEFAULT_BATCHING.stopCheckMode
    && normalizeReadbackInterval(batching.readbackInterval) === GLOBAL_DEFAULT_BATCHING.readbackInterval
    && normalizeReadbackInterval(batching.ringTokens) === GLOBAL_DEFAULT_BATCHING.ringTokens
    && normalizeReadbackInterval(batching.ringStop) === GLOBAL_DEFAULT_BATCHING.ringStop
    && normalizeReadbackInterval(batching.ringStaging) === GLOBAL_DEFAULT_BATCHING.ringStaging;
}

function isRuntimeGenerationAtGlobalDefaults(generation) {
  if (!generation || typeof generation !== 'object') {
    return false;
  }
  return (generation.disableCommandBatching === true) === GLOBAL_DEFAULT_GENERATION.disableCommandBatching;
}

function requireManifestDecodeLoopPositiveInt(value, label, modelId) {
  const normalized = normalizePositiveInt(value);
  if (normalized == null) {
    throw new Error(`Manifest "${modelId}" inference.sessionDefaults.decodeLoop.${label} must be a positive integer.`);
  }
  return normalized;
}

function requireManifestDecodeLoopStopCheckMode(value, modelId) {
  const normalized = normalizeStopCheckMode(value);
  if (normalized == null) {
    throw new Error(
      `Manifest "${modelId}" inference.sessionDefaults.decodeLoop.stopCheckMode must be "batch" or "per-token".`
    );
  }
  return normalized;
}

function buildManifestDecodeLoopRuntimePatch(manifest) {
  const decodeLoop = manifest?.inference?.sessionDefaults?.decodeLoop;
  if (decodeLoop == null) {
    return null;
  }
  const modelId = String(manifest?.modelId ?? 'unknown').trim() || 'unknown';
  if (typeof decodeLoop !== 'object') {
    throw new Error(
      `Manifest "${modelId}" inference.sessionDefaults.decodeLoop must be an object when provided.`
    );
  }
  const batchSize = requireManifestDecodeLoopPositiveInt(decodeLoop.batchSize, 'batchSize', modelId);
  const stopCheckMode = requireManifestDecodeLoopStopCheckMode(decodeLoop.stopCheckMode, modelId);
  const readbackInterval = requireManifestDecodeLoopPositiveInt(
    decodeLoop.readbackInterval,
    'readbackInterval',
    modelId
  );
  const disableCommandBatching = parseManifestDecodeLoopOptionalBoolean(
    decodeLoop.disableCommandBatching,
    'disableCommandBatching',
    modelId
  );

  const batchingPatch = {
    batchSize,
    stopCheckMode,
    readbackInterval,
  };
  const ringTokens = parseManifestDecodeLoopOptionalPositiveInt(
    decodeLoop.ringTokens,
    'ringTokens',
    modelId
  );
  if (ringTokens !== undefined) {
    batchingPatch.ringTokens = ringTokens;
  }
  const ringStop = parseManifestDecodeLoopOptionalPositiveInt(
    decodeLoop.ringStop,
    'ringStop',
    modelId
  );
  if (ringStop !== undefined) {
    batchingPatch.ringStop = ringStop;
  }
  const ringStaging = parseManifestDecodeLoopOptionalPositiveInt(
    decodeLoop.ringStaging,
    'ringStaging',
    modelId
  );
  if (ringStaging !== undefined) {
    batchingPatch.ringStaging = ringStaging;
  }

  return {
    batching: batchingPatch,
    generation: disableCommandBatching == null
      ? null
      : { disableCommandBatching: disableCommandBatching === true },
  };
}

export function applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, modelConfig) {
  void modelConfig;
  if (manifest?.inference?.schema === 'doppler.execution/v0') {
    return runtimeConfig;
  }
  const batching = runtimeConfig?.inference?.batching;
  const generation = runtimeConfig?.inference?.generation;
  const runtimeBatchingAtDefaults = isRuntimeBatchingAtGlobalDefaults(batching);
  const runtimeGenerationAtDefaults = isRuntimeGenerationAtGlobalDefaults(generation);

  const patch = buildManifestDecodeLoopRuntimePatch(manifest);
  if (!patch) {
    return runtimeConfig;
  }

  const runtimeDisableCommandBatching = generation?.disableCommandBatching === true;
  const manifestDisableCommandBatching = patch.generation?.disableCommandBatching === true;
  if (!runtimeBatchingAtDefaults) {
    throw new Error(
      'Manifest decodeLoop defaults cannot be merged after runtime batching overrides were already resolved. ' +
      'Set runtime.inference.batching explicitly to the desired final values, or remove manifest.inference.sessionDefaults.decodeLoop.'
    );
  }
  if (patch.generation && !runtimeGenerationAtDefaults && runtimeDisableCommandBatching !== manifestDisableCommandBatching) {
    throw new Error(
      'Manifest decodeLoop.disableCommandBatching conflicts with runtime.inference.generation.disableCommandBatching. ' +
      'Choose one explicit source of truth.'
    );
  }

  const nextRuntimeConfig = mergeRuntimeValues(runtimeConfig, {
    inference: {
      batching: patch.batching,
      ...(patch.generation ? { generation: patch.generation } : {}),
    },
  });
  log.info(
    'Pipeline',
    `Manifest decodeLoop applied (${manifest?.modelId ?? 'unknown'}): ` +
    `batchSize=${patch.batching.batchSize}, stopCheckMode=${patch.batching.stopCheckMode}, ` +
    `readbackInterval=${patch.batching.readbackInterval}, ` +
    `disableCommandBatching=${patch.generation?.disableCommandBatching === true}`
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
      } else if (requirement === 'subgroups-f16') {
        supported = hasSubgroups && hasF16;
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

function isGlobalKernelPathDtypeDefault(currentValue, key) {
  if (currentValue == null) {
    return true;
  }
  return currentValue === GLOBAL_DEFAULT_KERNEL_PATH_DTYPES[key];
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
      `runtime.inference.kvcache.kvDtype=${current.kvDtype ?? 'unset'} ` +
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
    `inference.defaultKernelPath="${resolvedKernelPath.id}" uses activationDtype=${kernelActivation}. ` +
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

function applyKernelPathRuntimeDtypeContract(resolvedKernelPath, runtimeConfig, kernelPathSource, modelId) {
  const contract = buildKernelPathDtypeContract(resolvedKernelPath);
  if (!contract) {
    return runtimeConfig;
  }

  const current = {
    activationDtype: normalizeKernelDtype(runtimeConfig.inference?.compute?.activationDtype),
    kvDtype: normalizeKernelDtype(runtimeConfig.inference?.kvcache?.kvDtype),
    outputDtype: normalizeKernelDtype(runtimeConfig.inference?.session?.compute?.defaults?.outputDtype),
  };
  const mismatches = describeKernelPathDtypeMismatch(contract, current);
  if (mismatches.length === 0) {
    return runtimeConfig;
  }

  if (kernelPathSource === 'config') {
    throw new Error(
      `KernelPath "${resolvedKernelPath?.id ?? 'unknown'}" selected from ${kernelPathSource} ` +
      `requires explicit matching runtime dtypes for "${modelId}". ` +
      `Mismatches: ${mismatches.join('; ')}. ` +
      'Set runtime.inference.compute.activationDtype, runtime.inference.kvcache.kvDtype, ' +
      'and runtime.inference.session.compute.defaults.outputDtype to match the kernel path.'
    );
  }

  const canApplyManifestDefaults = (
    (contract.activationDtype == null || isGlobalKernelPathDtypeDefault(current.activationDtype, 'activationDtype'))
    && (contract.kvDtype == null || isGlobalKernelPathDtypeDefault(current.kvDtype, 'kvDtype'))
    && (contract.outputDtype == null || isGlobalKernelPathDtypeDefault(current.outputDtype, 'outputDtype'))
  );
  if (!canApplyManifestDefaults) {
    throw new Error(
      `Manifest/model kernelPath "${resolvedKernelPath?.id ?? 'unknown'}" for "${modelId}" ` +
      `conflicts with runtime dtype overrides. Mismatches: ${mismatches.join('; ')}. ` +
      'Either remove the runtime dtype override or set it to match the kernel path.'
    );
  }

  const nextInference = {
    ...runtimeConfig.inference,
    compute: { ...runtimeConfig.inference.compute },
    kvcache: { ...runtimeConfig.inference.kvcache },
  };
  const dtypeChanges = [];

  if (contract.activationDtype && current.activationDtype !== contract.activationDtype) {
    nextInference.compute.activationDtype = contract.activationDtype;
    dtypeChanges.push(`activation=${current.activationDtype ?? 'unset'}->${contract.activationDtype}`);
  }

  if (contract.kvDtype && current.kvDtype !== contract.kvDtype) {
    nextInference.kvcache.kvDtype = contract.kvDtype;
    dtypeChanges.push(`kv=${current.kvDtype ?? 'unset'}->${contract.kvDtype}`);
  }

  if (contract.outputDtype && current.outputDtype !== contract.outputDtype) {
    nextInference.session = {
      ...(nextInference.session ?? {}),
      compute: {
        ...(nextInference.session?.compute ?? {}),
        defaults: {
          ...(nextInference.session?.compute?.defaults ?? {}),
          outputDtype: contract.outputDtype,
        },
      },
    };
    dtypeChanges.push(`session.outputDtype=${current.outputDtype ?? 'unset'}->${contract.outputDtype}`);
  }

  log.info(
    'Pipeline',
    `KernelPath ${resolvedKernelPath?.id ?? 'unknown'} applied manifest/model runtime dtype defaults: ${dtypeChanges.join(', ')}`
  );
  return { ...runtimeConfig, inference: nextInference };
}

export function resolveKernelPathState(options) {
  const {
    manifest,
    runtimeConfig,
    modelConfig,
    kernelCapabilities = null,
  } = options;

  log.debug(
    'Pipeline',
    `kernelPath sources: config=${runtimeConfig.inference.kernelPath}, model=${modelConfig.kernelPath}`
  );

  const configuredKernelPathRef = runtimeConfig.inference.kernelPath
    ?? modelConfig.kernelPath;
  let kernelPathSource = 'none';
  let resolvedKernelPath = null;
  const kernelPathPolicy = resolveKernelPathPolicy(runtimeConfig?.inference?.kernelPathPolicy);

  if (configuredKernelPathRef) {
    kernelPathSource = resolveKernelPathSource(
      runtimeConfig.inference.kernelPath,
      runtimeConfig.inference.kernelPathSource,
      modelConfig.kernelPath
    );
    const capabilities = kernelCapabilities && typeof kernelCapabilities === 'object'
      ? kernelCapabilities
      : getKernelCapabilitiesSafe();
    const effectiveKernelPathRef = resolveCapabilityKernelPathRef(
      configuredKernelPathRef,
      kernelPathSource,
      capabilities,
      kernelPathPolicy
    );
    if (effectiveKernelPathRef !== configuredKernelPathRef) {
      log.info(
        'Pipeline',
        `KernelPath auto-select: ${configuredKernelPathRef} -> ${effectiveKernelPathRef} ` +
        `(source=${kernelPathSource}, mode=${kernelPathPolicy.mode}, subgroups=${capabilities?.hasSubgroups === true})`
      );
    }
    try {
      resolvedKernelPath = resolveKernelPath(effectiveKernelPathRef);
    } catch (e) {
      throw new Error(`KernelPath resolution failed for '${effectiveKernelPathRef}': ${ (e).message}`);
    }

    assertKernelPathFeatureCompatibility(
      configuredKernelPathRef,
      effectiveKernelPathRef,
      resolvedKernelPath,
      kernelPathSource,
      capabilities,
      kernelPathPolicy
    );

    const stats = getKernelPathStats(resolvedKernelPath);
    log.info(
      'Pipeline',
      `KernelPath: ${resolvedKernelPath.id} (${stats.decodeSteps} decode steps, ${stats.uniqueKernels} kernels, source=${kernelPathSource})`
    );
    assertManifestKernelPathDtypeCompatibility(manifest, resolvedKernelPath, kernelPathSource);
  } else {
    log.info('Pipeline', 'KernelPath: none (no kernel path configured)');
  }

  const nextRuntimeConfig = applyKernelPathRuntimeDtypeContract(
    resolvedKernelPath,
    runtimeConfig,
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
