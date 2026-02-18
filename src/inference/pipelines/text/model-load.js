import { markWarmed as markKernelCacheWarmed } from '../../../gpu/kernel-selection-cache.js';
import { getKernelCapabilities } from '../../../gpu/device.js';
import { log } from '../../../debug/index.js';
import { setRuntimeConfig } from '../../../config/runtime.js';
import { resolvePreset } from '../../../config/loader.js';
import {
  resolveKernelPath,
  getKernelPathStats,
  getKernelPathActivationDtype,
  getKernelPathKVDtype,
  setActiveKernelPath,
  applyKernelOverrides,
} from '../../../config/kernel-path-loader.js';
import { autoTuneKernels, prewarmKernels } from '../../../gpu/kernels/index.js';
import { resolveCapabilityKernelPathRef } from './kernel-path-auto-select.js';
import { initTokenizer } from './init.js';

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

function resolveKernelPathSource(runtimeKernelPath, runtimeConfigKernelPath, modelKernelPath) {
  if (runtimeKernelPath) return 'runtime';
  if (runtimeConfigKernelPath) return 'config';
  if (modelKernelPath) return 'model';
  return 'manifest';
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
  const kernelPathKVDtype = getKernelPathKVDtype(resolvedKernelPath);
  if (!kernelPathActivationDtype && !kernelPathKVDtype) {
    return runtimeConfig;
  }

  const currentActivation = runtimeConfig.inference.compute.activationDtype;
  const currentKV = runtimeConfig.inference.kvcache.kvDtype;
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

  if (dtypeChanges.length === 0) {
    return runtimeConfig;
  }

  log.info(
    'Pipeline',
    `KernelPath ${resolvedKernelPath?.id ?? 'unknown'} runtime dtype overrides: ${dtypeChanges.join(', ')}`
  );
  return setRuntimeConfig({ ...runtimeConfig, inference: nextInference });
}

export function resolveAndActivateKernelPath(options) {
  const {
    manifest,
    runtimeKernelPath,
    runtimeConfig,
    modelConfig,
  } = options;

  log.debug(
    'Pipeline',
    `kernelPath sources: runtime=${runtimeKernelPath}, config=${runtimeConfig.inference.kernelPath}, model=${modelConfig.kernelPath}`
  );

  const configuredKernelPathRef = runtimeKernelPath
    ?? runtimeConfig.inference.kernelPath
    ?? modelConfig.kernelPath;
  let kernelPathSource = 'none';
  let resolvedKernelPath = null;

  if (configuredKernelPathRef) {
    kernelPathSource = resolveKernelPathSource(
      runtimeKernelPath,
      runtimeConfig.inference.kernelPath,
      modelConfig.kernelPath
    );
    const capabilities = getKernelCapabilitiesSafe();
    const effectiveKernelPathRef = resolveCapabilityKernelPathRef(
      configuredKernelPathRef,
      kernelPathSource,
      capabilities
    );
    if (effectiveKernelPathRef !== configuredKernelPathRef) {
      log.info(
        'Pipeline',
        `KernelPath auto-select: ${configuredKernelPathRef} -> ${effectiveKernelPathRef} (source=${kernelPathSource}, subgroups=${capabilities?.hasSubgroups === true})`
      );
    }
    try {
      resolvedKernelPath = resolveKernelPath(effectiveKernelPathRef);
      if (runtimeConfig.inference.kernelOverrides) {
        log.info('Pipeline', 'Applying kernel path overrides', runtimeConfig.inference.kernelOverrides);
        resolvedKernelPath = applyKernelOverrides(
          resolvedKernelPath,
          runtimeConfig.inference.kernelOverrides
        );
      }
      const stats = getKernelPathStats(resolvedKernelPath);
      log.info(
        'Pipeline',
        `KernelPath: ${resolvedKernelPath.id} (${stats.decodeSteps} decode steps, ${stats.uniqueKernels} kernels, source=${kernelPathSource})`
      );
    } catch (e) {
      resolvedKernelPath = null;
      log.warn('Pipeline', `Failed to resolve kernel path '${effectiveKernelPathRef}': ${ (e).message}`);
    }
  } else {
    log.info('Pipeline', 'KernelPath: none (no kernel path configured)');
  }

  setActiveKernelPath(resolvedKernelPath, kernelPathSource);
  const nextRuntimeConfig = applyKernelPathRuntimeDtypeOverrides(resolvedKernelPath, runtimeConfig);
  return {
    resolvedKernelPath,
    kernelPathSource,
    runtimeConfig: nextRuntimeConfig,
  };
}

export async function initTokenizerFromManifestPreset(manifest, baseUrl) {
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
  });
}
