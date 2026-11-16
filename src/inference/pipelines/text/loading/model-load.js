import { markWarmed as markKernelCacheWarmed } from "../../../../gpu/kernel-selection-cache.js";
import { getKernelCapabilities } from "../../../../gpu/device.js";
import { log } from "../../../../debug/index.js";
import { resolveKernelPath, getKernelPathStats, getKernelPathActivationDtype, getKernelPathOutputDtype, getKernelPathKVDtype, setActiveKernelPath } from "../../../../config/kernel-path-loader.js";
import { prewarmKernels } from "../../../../gpu/kernels/kernel-prewarm.js";
import { getKernelRegistry } from "../../../../gpu/kernels/kernel-configs.js";
import { initTokenizer } from "../init.js";
import { selectRuleValue } from "../../../../rules/rule-registry.js";
import { mergeRuntimeValues } from "../../../../config/runtime-merge.js";
import { mergeKernelPathPolicy } from "../../../../config/merge/kernel-path-policy.js";
import { resolveRuntimeActivationDtype, buildSessionRuntimePatch } from "../execution/execution-runtime-builders.js";
export { applyModelBatchingRuntimeDefaults } from "./decode-session-config.js";

function validateKernelWarmupMode(mode) {
  if (mode !== 'parallel' && mode !== 'sequential') {
    throw new Error(
      `runtime.shared.kernelWarmup.prewarmMode must be "parallel" or "sequential"; got "${mode}".`
    );
  }
}

export async function runKernelWarmup(options) {
  const { useGPU, kernelWarmup } = options;
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

function buildKernelRequiredFeaturesByShaderEntry(registry) {
  const index = new Map();
  for (const variants of Object.values(registry.configs)) {
    for (const config of Object.values(variants)) {
      const shaderFile = normalizeKernelFileName(config.shaderFile);
      if (!shaderFile) continue;
      const key = `${shaderFile}#${config.entryPoint}`;
      const requires = index.get(key) ?? new Set();
      for (const requirement of config.requires) requires.add(requirement);
      index.set(key, requires);
    }
  }
  return index;
}

const requiredFeaturesByRegistry = new WeakMap();

function getKernelRequiredFeaturesByShaderEntry(registry) {
  const cached = requiredFeaturesByRegistry.get(registry);
  if (cached) return cached;
  const index = buildKernelRequiredFeaturesByShaderEntry(registry);
  requiredFeaturesByRegistry.set(registry, index);
  return index;
}

function collectKernelPathSteps(kernelPath) {
  const steps = [];
  const append = (list) => {
    for (const step of list ?? []) if (step && typeof step === 'object') steps.push(step);
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

function findKernelPathUnsupportedFeatureUsages(kernelPath, capabilities, registry) {
  const offenders = [];
  const seen = new Set();
  const requiredFeatures = getKernelRequiredFeaturesByShaderEntry(registry);
  const hasSubgroups = capabilities?.hasSubgroups === true;
  const hasF16 = capabilities?.hasF16 === true;
  for (const step of collectKernelPathSteps(kernelPath)) {
    const kernelFile = normalizeKernelFileName(step.kernel);
    if (!kernelFile) continue;
    const entryPoint = String(step.entry ?? 'main').trim() || 'main';
    const key = `${kernelFile}#${entryPoint}`;
    const requirements = requiredFeatures.get(key);
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
  resolvedKernelPath,
  kernelPathSource,
  capabilities,
  kernelPathPolicy,
  registry
) {
  const unsupportedUsages = findKernelPathUnsupportedFeatureUsages(resolvedKernelPath, capabilities, registry);
  if (unsupportedUsages.length === 0) return;

  const sourceScope = kernelPathPolicy.sourceScope ?? kernelPathPolicy.allowSources ?? [];
  const policyAllowsSource = kernelPathPolicy.mode === 'capability-aware'
    && sourceScope.includes(kernelPathSource);
  const remapRequested = policyAllowsSource && kernelPathPolicy.onIncompatible === 'remap';
  const summary = summarizeUnsupportedKernelUsages(unsupportedUsages);

  if (remapRequested) {
    throw new Error(
      `KernelPath "${resolvedKernelPath?.id ?? 'unknown'}" requires unsupported GPU features (${summary}) ` +
      `for source "${kernelPathSource}". String registry remaps are removed; use execution graph ` +
      'capability transforms or choose a compatible inline kernelPath.'
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
      `runtime.inference.session.compute.defaults.activationDtype=${current.activationDtype ?? 'unset'} ` +
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

export function assertManifestComputeLaneBinding({ manifest, runtimeConfig }) {
  const declared = normalizeKernelDtype(manifest?.quantizationInfo?.compute);
  if (!declared) return;

  const session = runtimeConfig?.inference?.session ?? {};
  const computeDefaults = session.compute?.defaults ?? {};
  const candidates = [
    ['session.compute.defaults.activationDtype', computeDefaults.activationDtype],
    ['session.compute.defaults.mathDtype', computeDefaults.mathDtype],
    ['session.compute.defaults.accumDtype', computeDefaults.accumDtype],
  ];

  const mismatches = [];
  for (const [field, value] of candidates) {
    const resolved = normalizeKernelDtype(value);
    if (resolved && resolved !== declared) {
      mismatches.push(`${field}=${value}`);
    }
  }
  if (mismatches.length === 0) return;

  throw new Error(
    `Manifest "${manifest?.modelId ?? 'unknown'}" declares ` +
    `quantizationInfo.compute=${declared} but runtime resolved ` +
    `[${mismatches.join('; ')}]. ` +
    'The manifest variant tag is the lane identity — load the manifest variant ' +
    'whose compute lane matches the runtime profile, or pick a runtime profile ' +
    'whose dtype defaults match this manifest.'
  );
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

function getExplicitRuntimeDtypeOverrides(runtimeOverrides, executionV1State, resolvedKernelPath, kernelPathSource) {
  const inference = runtimeOverrides?.inference;
  // A selective lane consumes the legacy compute field as its request; the
  // compiler's declared session still owns the residual activation dtype.
  // Canonical session overrides remain strict, as do untransformed paths.
  const selectiveRequest = inference?.session?.compute?.defaults?.activationDtype === undefined
    && inference?.compute?.activationDtype === 'f16'
    && kernelPathSource === 'execution-v1'
    && executionV1State?.laneIntegrity?.policy?.kind === 'explicit-lane'
    && executionV1State.laneIntegrity.policy.dtypeEffect === 'selective-f16'
    && executionV1State.appliedTransforms?.length > 0
    && executionV1State.runtimeInferencePatch?.kernelPath?.id === resolvedKernelPath?.id;
  return {
    activationDtype: selectiveRequest
      ? resolveRuntimeActivationDtype(executionV1State.runtimeInferencePatch)
      : resolveRuntimeActivationDtype(inference),
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
  modelId,
  executionV1State
) {
  const contract = buildKernelPathDtypeContract(resolvedKernelPath);
  if (!contract) {
    return runtimeConfig;
  }

  const explicitRuntime = getExplicitRuntimeDtypeOverrides(runtimeOverrides, executionV1State, resolvedKernelPath, kernelPathSource);
  const current = {
    activationDtype: resolveRuntimeActivationDtype(runtimeConfig.inference),
    kvDtype: normalizeKernelDtype(runtimeConfig.inference?.session?.kvcache?.kvDtype),
    outputDtype: normalizeKernelDtype(runtimeConfig.inference?.session?.compute?.defaults?.outputDtype),
  };
  const mismatches = describeKernelPathDtypeMismatch(contract, current);
  const explicitMismatches = describeExplicitRuntimeDtypeMismatch(contract, explicitRuntime);
  if (mismatches.length === 0 && explicitMismatches.length === 0) {
    if (runtimeConfig.inference?.session?.compute?.defaults?.activationDtype !== undefined
      && runtimeConfig.inference?.compute?.activationDtype !== current.activationDtype) {
      return mergeRuntimeValues(runtimeConfig, { inference: buildSessionRuntimePatch(
        { compute: { defaults: { activationDtype: current.activationDtype } } },
        { includeDecodeLoop: false }
      ) });
    }
    return runtimeConfig;
  }

  if (kernelPathSource === 'config') {
    throw new Error(
      `KernelPath "${resolvedKernelPath?.id ?? 'unknown'}" selected from ${kernelPathSource} ` +
      `requires explicit matching runtime dtypes for "${modelId}". ` +
      `Mismatches: ${[...mismatches, ...explicitMismatches].join('; ')}. ` +
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
  const kernelRegistry = getKernelRegistry();
  const {
    manifest,
    runtimeConfig,
    runtimeOverrides = null,
    modelConfig,
    executionV1State = null,
    kernelCapabilities = null,
  } = options;

  log.debug(
    'Pipeline',
    `kernelPath sources: config=${runtimeConfig.inference.kernelPath}, model=${modelConfig.kernelPath}`
  );

  const configuredKernelPathRef = runtimeConfig.inference.kernelPath
    ?? modelConfig.kernelPath
    ?? null;
  let kernelPathSource = 'none';
  let resolvedKernelPath = null;
  const kernelPathPolicy = mergeKernelPathPolicy(undefined, runtimeConfig?.inference?.kernelPathPolicy);

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
        resolvedKernelPath,
        kernelPathSource,
        capabilities,
        kernelPathPolicy,
        kernelRegistry
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
    String(manifest?.modelId ?? 'unknown').trim() || 'unknown',
    executionV1State
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

export async function initTokenizerFromManifest(manifest, baseUrl, storageContext = null) {
  return initTokenizer(manifest, {
    baseUrl: baseUrl ?? undefined,
    tokenizerHints: null,
    storageContext: storageContext ?? undefined,
  });
}
