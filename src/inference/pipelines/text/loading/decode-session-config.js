import { getKernelCapabilities } from "../../../../gpu/device.js";
import { log } from "../../../../debug/index.js";
import { mergeRuntimeValues } from "../../../../config/runtime-merge.js";
import { applyModelSessionDeclaration } from "../../../../config/model-session-declarations.js";
import { READBACK_MODES } from "../../../../config/schema/execution/execution-v1.schema.js";

function normalizePositiveInt(value) {
  if (!Number.isFinite(value)) return null;
  const normalized = Math.floor(value);
  return normalized >= 1 ? normalized : null;
}

function normalizeStopCheckMode(value) {
  if (value === 'batch' || value === 'per-token') return value;
  return null;
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

function resolveDecodeLoopRuntimeSession(runtimeConfig, runtimeOverrides, useExplicitRuntimeOverrides) {
  const runtimeSession = runtimeConfig?.inference?.session ?? null;
  if (runtimeConfig?.inference?.kernelPathSource === 'execution-v1') {
    return runtimeSession;
  }
  if (!useExplicitRuntimeOverrides) {
    return runtimeSession;
  }
  const inferenceOverrides = runtimeOverrides?.inference;
  if (!inferenceOverrides || typeof inferenceOverrides !== 'object' || Array.isArray(inferenceOverrides)) {
    return null;
  }
  if (!Object.prototype.hasOwnProperty.call(inferenceOverrides, 'session')) {
    return null;
  }
  return inferenceOverrides.session ?? null;
}

function buildResolvedDecodeLoopRuntimePatch(
  runtimeConfig,
  manifest,
  runtimeOverrides = undefined,
  useExplicitRuntimeOverrides = false
) {
  const runtimeSession = resolveDecodeLoopRuntimeSession(
    runtimeConfig,
    runtimeOverrides,
    useExplicitRuntimeOverrides
  );
  // The manifest session, with the per-model declaration
  // (config/runtime/model-sessions.json) filling a KV length the manifest does
  // not declare; runtime profile/override values still win field by field.
  const resolvedSession = mergeRuntimeValues(
    applyModelSessionDeclaration(manifest),
    runtimeSession ?? {}
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
  const maxBatchDecodeTokens = parseSessionDecodeLoopOptionalPositiveInt(
    decodeLoop.maxBatchDecodeTokens,
    'maxBatchDecodeTokens',
    modelId
  );
  if (decodeLoop.layerSubmitInterval != null && !Number.isSafeInteger(decodeLoop.layerSubmitInterval)) {
    throw new Error(`Manifest "${modelId}" inference.session.decodeLoop.layerSubmitInterval must be a positive integer or null.`);
  }
  const layerSubmitInterval = parseSessionDecodeLoopOptionalPositiveInt(
    decodeLoop.layerSubmitInterval,
    'layerSubmitInterval',
    modelId
  );

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
        ...(maxBatchDecodeTokens !== undefined ? { maxBatchDecodeTokens } : {}),
        ...(layerSubmitInterval !== undefined ? { layerSubmitInterval } : {}),
        ...(ringTokens !== undefined ? { ringTokens } : {}),
        ...(ringStop !== undefined ? { ringStop } : {}),
        ...(ringStaging !== undefined ? { ringStaging } : {}),
        ...(disableCommandBatching !== undefined ? { disableCommandBatching } : {}),
      },
    },
    batching: batchingPatch,
  };
}

export function applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, modelConfig, runtimeOverrides = undefined) {
  void modelConfig;
  const patch = buildResolvedDecodeLoopRuntimePatch(
    runtimeConfig,
    manifest,
    runtimeOverrides,
    runtimeOverrides != null
  );
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

  // Promote the resolved session and manifest largeWeights into runtimeConfig
  // so existing pipeline reads of getRuntimeConfig().inference.session.* and
  // .largeWeights.* pick up per-model values. buildResolvedDecodeLoopRuntimePatch()
  // already applied runtime-over-manifest session precedence field-by-field.
  // For largeWeights, an explicit runtime array wins, including [] to disable
  // manifest gpu-resident overrides on constrained verification hosts.
  const manifestInf = manifest?.inference ?? {};
  const manifestLargeWeights = manifestInf.largeWeights;
  const runtimeLargeWeightsOverrides =
    runtimeConfig?.inference?.largeWeights?.gpuResidentOverrides;
  const hasRuntimeLargeWeightsOverride = Array.isArray(runtimeLargeWeightsOverrides);
  const largeWeightsPatch = (!hasRuntimeLargeWeightsOverride
    && manifestLargeWeights && typeof manifestLargeWeights === 'object'
    && Array.isArray(manifestLargeWeights.gpuResidentOverrides)
    && manifestLargeWeights.gpuResidentOverrides.length > 0)
    ? { gpuResidentOverrides: manifestLargeWeights.gpuResidentOverrides }
    : null;

  const nextRuntimeConfig = mergeRuntimeValues(runtimeConfig, {
    inference: {
      session: patch.session,
      batching: patch.batching,
      ...(largeWeightsPatch ? { largeWeights: largeWeightsPatch } : {}),
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
