import { markWarmed as markKernelCacheWarmed } from '../../../gpu/kernel-selection-cache.js';
import { getKernelCapabilities } from '../../../gpu/device.js';
import { log } from '../../../debug/index.js';
import { resolvePreset } from '../../../config/loader.js';
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

function validateKernelWarmupMode(mode) {
  if (mode !== 'parallel' && mode !== 'sequential') {
    throw new Error(
      `runtime.shared.kernelWarmup.prewarmMode must be "parallel" or "sequential"; got "${mode}".`
    );
  }
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
  if (normalized === 'execution_v0') return 'execution-v0';
  if (normalized === 'execution-v0') return 'execution-v0';
  return normalized || 'none';
}

function resolveKernelPathSource(runtimeConfigKernelPath, runtimeKernelPathSourceHint, modelKernelPath) {
  if (runtimeConfigKernelPath) {
    const sourceHint = normalizeKernelPathSourceHint(runtimeKernelPathSourceHint);
    if (sourceHint === 'execution-v0') return 'execution-v0';
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

  if (kernelPathSource === 'execution-v0' && typeof effectiveKernelPathRef !== 'string') {
    const remediation = policyAllowsSource
      ? 'Execution-v0 inline kernel paths are not auto-remapped yet. Use subgroup/f16-compatible execution steps, or set runtime.inference.kernelPath to a compatible string preset (for example "gemma2-q4k-dequant-f32a").'
      : 'Enable runtime.inference.kernelPathPolicy.sourceScope to include "execution-v0", then use compatible execution steps or a compatible preset id.';
    throw new Error(
      `[ExecutionV0] Inline kernelPath requires unsupported GPU features. ` +
      `Offending steps: ${summary}. ${remediation}`
    );
  }

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

function applyKernelPathRuntimeDtypeOverrides(resolvedKernelPath, runtimeConfig) {
  const kernelPathActivationDtype = getKernelPathActivationDtype(resolvedKernelPath);
  const kernelPathOutputDtype = getKernelPathOutputDtype(resolvedKernelPath) ?? kernelPathActivationDtype;
  const kernelPathKVDtype = getKernelPathKVDtype(resolvedKernelPath);
  if (!kernelPathActivationDtype && !kernelPathOutputDtype && !kernelPathKVDtype) {
    return runtimeConfig;
  }

  const currentActivation = runtimeConfig.inference.compute.activationDtype;
  const currentKV = runtimeConfig.inference.kvcache.kvDtype;
  const currentOutput = runtimeConfig.inference?.session?.compute?.defaults?.outputDtype;
  const nextInference = {
    ...runtimeConfig.inference,
    compute: { ...runtimeConfig.inference.compute },
    kvcache: { ...runtimeConfig.inference.kvcache },
  };
  const dtypeChanges = [];

  if (kernelPathActivationDtype && currentActivation !== kernelPathActivationDtype) {
    nextInference.compute.activationDtype = kernelPathActivationDtype;
    dtypeChanges.push(`activation=${currentActivation}->${kernelPathActivationDtype}`);
  }

  if (kernelPathKVDtype && currentKV !== kernelPathKVDtype) {
    nextInference.kvcache.kvDtype = kernelPathKVDtype;
    dtypeChanges.push(`kv=${currentKV}->${kernelPathKVDtype}`);
  }

  if (kernelPathOutputDtype && currentOutput !== kernelPathOutputDtype) {
    nextInference.session = {
      ...(nextInference.session ?? {}),
      compute: {
        ...(nextInference.session?.compute ?? {}),
        defaults: {
          ...(nextInference.session?.compute?.defaults ?? {}),
          outputDtype: kernelPathOutputDtype,
        },
      },
    };
    dtypeChanges.push(`session.outputDtype=${currentOutput ?? 'undefined'}->${kernelPathOutputDtype}`);
  }

  if (dtypeChanges.length === 0) {
    return runtimeConfig;
  }

  log.info(
    'Pipeline',
    `KernelPath ${resolvedKernelPath?.id ?? 'unknown'} runtime dtype overrides: ${dtypeChanges.join(', ')}`
  );
  return { ...runtimeConfig, inference: nextInference };
}

export function resolveKernelPathState(options) {
  const {
    manifest,
    runtimeConfig,
    modelConfig,
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
    const capabilities = getKernelCapabilitiesSafe();
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

  const nextRuntimeConfig = applyKernelPathRuntimeDtypeOverrides(resolvedKernelPath, runtimeConfig);
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

export async function initTokenizerFromManifestPreset(manifest, baseUrl, storageContext = null) {
  const presetId = manifest.inference?.presetId;
  if (!presetId) {
    throw new Error(
      `Manifest "${manifest.modelId ?? 'unknown'}" is missing inference.presetId. ` +
      'Re-convert the model using the latest converter.'
    );
  }
  const preset = resolvePreset(presetId);
  return initTokenizer(manifest, {
    baseUrl: baseUrl ?? undefined,
    presetTokenizer: preset?.tokenizer,
    storageContext: storageContext ?? undefined,
  });
}
