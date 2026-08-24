import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { runNodeCommand } from '../../tooling/node-command-runner.js';
import { runBrowserCommandInNode } from '../../tooling/node-browser-command-runner.js';
import { writeProgramBundle } from '../../tooling/program-bundle.js';
import {
  TOOLING_COMMANDS,
  normalizeToolingCommandRequest,
} from '../../tooling/command-api.js';
import { createToolingErrorEnvelope } from '../../tooling/command-envelope.js';
import {
  asStringOrNull,
  resolveBrowserModelUrl,
  resolveNodeModelUrl,
  resolveStaticRootDir,
  resolveRdrrRoot,
} from '../cli-model-resolution.js';
import { isPlainObject } from '../../formats/plain-object.js';
import {
  toSummary,
  formatNumber,
  formatMs,
  saveBenchResult,
  loadBaseline,
  compareBenchResults,
  printManifestSummary,
  printDeviceInfo,
  printConvertContractSummary,
  printConvertReportSummary,
  printMetricsSummary,
} from '../cli-output.js';
import { formatRuntimeProfiles, listRuntimeProfiles } from '../runtime-profiles.js';
import { persistBrowserRelayReport } from '../browser-report-output.js';
import { isAbsoluteUrl, normalizeModelUrl, parseJsonObjectFlag, performIntake, readJsonObjectFile, readJsonObjectInput, runBoundaryCommand, runBundleCommand, runCommandOnSurface } from '../commands/bundle.js';

export const NODE_WEBGPU_INCOMPLETE_MESSAGE = 'node command: WebGPU runtime is incomplete in Node';

export const DEFAULT_CLI_POLICY = {
  defaults: {
    surface: {
      default: 'auto',
      allowed: ['auto', 'node', 'browser'],
    },
    bench: {
      cacheMode: 'warm',
    },
    cacheMode: null,
    loadMode: null,
    benchmark: {
      saveDir: './benchmarks/vendors/results',
    },
  },
  surfaceFallback: {
    enabled: true,
    from: 'auto',
    to: 'browser',
    errorFragments: [NODE_WEBGPU_INCOMPLETE_MESSAGE],
  },
};

export function parseRuntimeConfigUrl(value) {
  const normalized = asStringOrNull(value);
  if (normalized === null) return null;
  return isAbsoluteUrl(normalized)
    ? normalized
    : pathToFileURL(path.resolve(normalized)).href;
}

export function parseSurface(value, command, policy = DEFAULT_CLI_POLICY) {
  const normalizedInput = asStringOrNull(value);
  const normalizedSurface = policy.defaults && policy.defaults.surface && policy.defaults.surface.default
    ? policy.defaults.surface.default
    : 'auto';
  const allowedSurfaces = policy.defaults && Array.isArray(policy.defaults.surface?.allowed)
    ? policy.defaults.surface.allowed
    : ['auto', 'node', 'browser'];
  const normalized = String(normalizedInput === null ? normalizedSurface : normalizedInput).trim().toLowerCase();
  if (!allowedSurfaces.includes(normalized)) {
    throw new Error('--surface must be one of auto, node, browser');
  }
  if ((command === 'convert' || command === 'refresh-integrity' || command === 'release') && normalized === 'browser') {
    throw new Error(`${command} is not supported on browser relay. Use --surface node or --surface auto.`);
  }
  if ((command === 'diagnose' || command === 'lora' || command === 'distill') && normalized === 'browser') {
    throw new Error(`${command} is not supported on browser relay. Use --surface node or --surface auto.`);
  }
  return normalized;
}

export const CONFIG_ENVELOPE_KNOWN_KEYS = new Set([
  'request',
  'run',
  'runtimeProfile',
  'runtimeConfigUrl',
  'runtimeConfig',
]);

export function buildRuntimeOverridesFromObject(source = {}, sourceLabel = '--config') {
  const normalizedRuntimeProfile = asStringOrNull(source.runtimeProfile);
  const normalizedRuntimeConfigUrl = asStringOrNull(source.runtimeConfigUrl) == null
    ? null
    : parseRuntimeConfigUrl(source.runtimeConfigUrl);

  const hasInlineRuntimeConfig = Object.prototype.hasOwnProperty.call(source, 'runtimeConfig');
  let runtimeConfig = null;
  if (hasInlineRuntimeConfig) {
    if (!isPlainObject(source.runtimeConfig)) {
      throw new Error(`${sourceLabel} runtimeConfig must be a JSON object when provided.`);
    }
    runtimeConfig = source.runtimeConfig;
  }

  return {
    runtimeProfile: normalizedRuntimeProfile,
    runtimeConfigUrl: normalizedRuntimeConfigUrl,
    runtimeConfig,
  };
}

export function resolveConfigEnvelope(configPayload) {
  if (!isPlainObject(configPayload)) {
    throw new Error('--config must resolve to a JSON object.');
  }
  if (configPayload.request !== undefined && !isPlainObject(configPayload.request)) {
    throw new Error('--config field "request" must be a JSON object when provided.');
  }
  if (configPayload.run !== undefined && !isPlainObject(configPayload.run)) {
    throw new Error('--config field "run" must be a JSON object when provided.');
  }
  if (configPayload.request !== undefined) {
    const topLevelUnknown = Object.keys(configPayload).filter(
      (key) => !CONFIG_ENVELOPE_KNOWN_KEYS.has(key)
    );
    if (topLevelUnknown.length > 0) {
      throw new Error(
        `--config has unknown top-level keys for request envelope: ${topLevelUnknown.join(', ')}.`
      );
    }
  }

  const hasRequestEnvelope = isPlainObject(configPayload.request);
  const requestPayload = hasRequestEnvelope
    ? { ...configPayload.request }
    : { ...configPayload };
  const topLevelRuntimeConfig = hasRequestEnvelope
    ? buildRuntimeOverridesFromObject(configPayload, '--config')
    : {
      runtimeProfile: null,
      runtimeConfigUrl: null,
      runtimeConfig: null,
    };
  const requestRuntimeConfig = buildRuntimeOverridesFromObject(requestPayload, '--config.request');
  if (hasRequestEnvelope) {
    if (
      topLevelRuntimeConfig.runtimeProfile != null
      && requestRuntimeConfig.runtimeProfile != null
    ) {
      throw new Error(
        'Cannot set runtimeProfile in both --config payload top-level and --config.request.'
      );
    }
    if (
      topLevelRuntimeConfig.runtimeConfigUrl != null
      && requestRuntimeConfig.runtimeConfigUrl != null
    ) {
      throw new Error(
        'Cannot set runtimeConfigUrl in both --config payload top-level and --config.request.'
      );
    }
    if (
      topLevelRuntimeConfig.runtimeConfig != null
      && requestRuntimeConfig.runtimeConfig != null
    ) {
      throw new Error(
        'Cannot set runtimeConfig in both --config payload top-level and --config.request.'
      );
    }
  }

  const requestInput = {
    ...requestPayload,
    runtimeProfile: (
      topLevelRuntimeConfig.runtimeProfile !== null
      ? topLevelRuntimeConfig.runtimeProfile
      : requestRuntimeConfig.runtimeProfile
    ),
    runtimeConfigUrl: (
      topLevelRuntimeConfig.runtimeConfigUrl !== null
      ? topLevelRuntimeConfig.runtimeConfigUrl
      : requestRuntimeConfig.runtimeConfigUrl
    ),
    runtimeConfig: (
      topLevelRuntimeConfig.runtimeConfig !== null
      ? topLevelRuntimeConfig.runtimeConfig
      : requestRuntimeConfig.runtimeConfig
    ),
  };
  return {
    request: requestInput,
    run: isPlainObject(configPayload.run) ? configPayload.run : {},
  };
}

export function resolveSurfaceForCommand(command, parsed, runConfig, policy = DEFAULT_CLI_POLICY) {
  const fromCli = asStringOrNull(parsed.flags.surface);
  const fromRun = asStringOrNull(runConfig?.surface);
  return parseSurface(fromCli ?? fromRun ?? null, command, policy);
}

export function isNodeWebGPUFallbackCandidate(error, fallbackPolicy = DEFAULT_CLI_POLICY.surfaceFallback) {
  const message = error?.message || String(error || '');
  const fallbackSignatures = Array.isArray(fallbackPolicy?.errorFragments) && fallbackPolicy.errorFragments.length > 0
    ? fallbackPolicy.errorFragments
    : [NODE_WEBGPU_INCOMPLETE_MESSAGE];
  return fallbackSignatures.some((signature) => message.includes(signature));
}

export function isTrainingCommandFlow(request) {
  if (!request || typeof request !== 'object') return false;
  if (request.workload === 'training') return true;
  if (request.command === 'lora' || request.command === 'distill') return true;
  return request.command === 'bench' && (request.workload === 'training' || request.workloadType === 'training');
}

export function resolveErrorSurface(error, fallbackSurface = null) {
  return (
    asStringOrNull(fallbackSurface)
    || asStringOrNull(error?.surface)
    || asStringOrNull(error?.details?.surface)
    || null
  );
}

export function createCliToolingErrorEnvelope(error, context = {}) {
  return createToolingErrorEnvelope(error, {
    surface: resolveErrorSurface(error, context.surface),
    request: context.request ?? null,
  });
}

export async function runWithAutoSurface(request, runConfig, jsonOutput, policy = DEFAULT_CLI_POLICY) {
  if (
    request.command === 'convert'
    || request.command === 'refresh-integrity'
    || request.command === 'release'
    || request.command === 'diagnose'
  ) {
    return runCommandOnSurface(request, 'node', runConfig, jsonOutput);
  }
  const fallbackPolicy = policy?.surfaceFallback || { enabled: false };

  try {
    return await runCommandOnSurface(request, 'node', runConfig, jsonOutput);
  } catch (error) {
    if (!fallbackPolicy.enabled || !isNodeWebGPUFallbackCandidate(error, fallbackPolicy)) {
      throw error;
    }
    if (isTrainingCommandFlow(request)) {
      const downgradeError = new Error(
        (request.command === 'lora' || request.command === 'distill')
          ? 'Training command auto-surface downgrade is blocked. Re-run with --surface node after fixing Node WebGPU support.'
          : 'Training command auto-surface downgrade is blocked. Re-run with --surface node after fixing Node WebGPU support, or explicitly choose --surface browser.'
      );
      downgradeError.code = 'training_surface_downgrade_blocked';
      downgradeError.surface = 'node';
      downgradeError.command = request.command;
      downgradeError.workload = request.workload;
      downgradeError.workloadType = request.workloadType || null;
      downgradeError.fromSurface = 'node';
      downgradeError.toSurface = fallbackPolicy.to || 'browser';
      throw downgradeError;
    }
    if (fallbackPolicy.to !== 'browser') {
      throw error;
    }
    if (!jsonOutput) {
      console.error('[surface] node WebGPU unavailable, falling back to browser');
    }
    return runCommandOnSurface(request, 'browser', runConfig, jsonOutput);
  }
}

export function mergeRunConfig(base, ...overrides) {
  const merged = isPlainObject(base) ? { ...base } : {};
  for (const source of overrides) {
    if (!isPlainObject(source)) {
      continue;
    }
    for (const [key, value] of Object.entries(source)) {
      if ((key === 'browser' || key === 'bench') && isPlainObject(value)) {
        merged[key] = {
          ...(isPlainObject(merged[key]) ? merged[key] : {}),
          ...value,
        };
        continue;
      }
      merged[key] = value;
    }
  }
  return merged;
}

export async function runManifestSweep(manifest, commandContext, jsonOutput, policy = DEFAULT_CLI_POLICY) {
  const defaults = manifest.defaults || {};
  const results = [];

  for (let i = 0; i < manifest.runs.length; i++) {
    const run = manifest.runs[i];
    const label = run.label || run.modelId || run.request?.modelId || `run-${i}`;
    if (!jsonOutput) {
      console.error(`[sweep] (${i + 1}/${manifest.runs.length}) ${label}`);
    }

    const requestInput = {
      ...commandContext.request,
      ...(isPlainObject(defaults.request) ? defaults.request : {}),
      ...(isPlainObject(run.request) ? run.request : {}),
      command: commandContext.request.command,
    };
    const modelId = asStringOrNull(run.modelId) || asStringOrNull(defaults.modelId);
    if (modelId) {
      requestInput.modelId = modelId;
    }
    const modelUrl = asStringOrNull(run.modelUrl) || asStringOrNull(defaults.modelUrl);
    if (modelUrl) {
      requestInput.modelUrl = modelUrl;
    }
    const runtimeProfile = asStringOrNull(run.runtimeProfile)
      || asStringOrNull(defaults.runtimeProfile);
    if (runtimeProfile) {
      requestInput.runtimeProfile = runtimeProfile;
    }

    const mergedRunConfig = mergeRunConfig(commandContext.runConfig, defaults.run, run.run);
    let request = null;
    let surface = commandContext.surface;
    try {
      request = normalizeToolingCommandRequest(requestInput);
      surface = commandContext.surfaceFromCli
        ? commandContext.surface
        : resolveSurfaceForCommand(
          request.command,
          { flags: { surface: null } },
          mergedRunConfig,
          policy
        );
      const response = surface === 'auto'
        ? await runWithAutoSurface(request, mergedRunConfig, jsonOutput, policy)
        : await runCommandOnSurface(request, surface, mergedRunConfig, jsonOutput);
      results.push({ label, response, error: null });
    } catch (error) {
      results.push({
        label,
        response: null,
        error: createCliToolingErrorEnvelope(error, {
          surface: surface === 'auto' ? null : surface,
          request,
        }),
      });
      if (!jsonOutput) {
        console.error(`[sweep] ${label} FAILED: ${error.message}`);
      }
    }
  }

  return results;
}
