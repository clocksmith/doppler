import { releasePerLayerInputBuffer } from "../../per-layer/per-layer-buffer.js";
import { getActivationRepresentation } from "../session/activation-storage.js";
import { getDevice } from "../../../../../gpu/device.js";
import { releaseBuffer } from "../../../../../memory/buffer-pool.js";
import { createCommandRecorder, createProfilingRecorder, CommandRecorder } from "../../../../../gpu/command-recorder.js";
import { log } from "../../../../../debug/index.js";
import { selectRuleValue } from "../../../../../rules/rule-registry.js";
import { isDecodeRecorderEnabled, isProfileDecodeRecorderEnabled } from "../../execution/execution-plan.js";
import { embed } from "../../embedding/embed.js";
import { resolvePerLayerInputsSession } from "../session/session-context.js";
import { processLayer, hasConvLayers } from "../../execution/layer.js";
import { isWeightBuffer, isCpuWeightBuffer, isGpuBufferInstance, isSplitWeightBuffer, getWeightDtype, getWeightMetadata } from "../../../../../gpu/weight-buffer.js";
import { preparePerLayerInputs, prefetchPerLayerRow } from "../../per-layer/per-layer-inputs.js";

export const UNKNOWN_TOKEN_TEXT = '<unknown>';

export function sumProfileTimings(timings) {
  if (!timings || Object.keys(timings).length === 0) return null;
  let total = 0;
  for (const value of Object.values(timings)) {
    if (Number.isFinite(value)) {
      total += value;
    }
  }
  return total;
}

export function getEffectiveActivationDtype(state, opts) {
  const executionPlanDtype = opts?.executionPlan?.activationDtype;
  if (executionPlanDtype !== undefined && executionPlanDtype !== null) {
    return executionPlanDtype;
  }
  if (executionPlanDtype === null) {
    throw new Error('[Pipeline] executionPlan.activationDtype is required when provided and cannot be null.');
  }
  return state.runtimeConfig.inference.compute.activationDtype;
}

export function getTokenTextOrUnknown(tokenizer, tokenId) {
  if (!tokenizer || typeof tokenizer.decode !== 'function') {
    return UNKNOWN_TOKEN_TEXT;
  }

  const tokenText = tokenizer.decode([tokenId], false, false);
  if (typeof tokenText !== 'string' || tokenText.length === 0) {
    return UNKNOWN_TOKEN_TEXT;
  }

  return tokenText;
}

export function isOwnedDecodeBuffer(candidate, decodeHiddenBuffer, decodeAltBuffer) {
  if (candidate === decodeHiddenBuffer) {
    return true;
  }
  return candidate === decodeAltBuffer;
}

export function schedulePlePrefetchForToken(state, tokenId) {
  if (state?.prefetchPleNextToken !== true) {
    return;
  }
  const config = state.modelConfig;
  const pleHiddenSize = Number(config?.hiddenSizePerLayerInput ?? 0);
  if (!Number.isFinite(pleHiddenSize) || pleHiddenSize <= 0) {
    return;
  }
  const pleWeights = state.weights.get('per_layer_inputs');
  if (!pleWeights?.embedTokensPerLayer) {
    return;
  }
  const resolvedPerLayerInputsSession = resolvePerLayerInputsSession(
    config.perLayerInputsSession ?? null,
    state.runtimeConfig?.inference?.session?.perLayerInputs ?? null
  );
  state.plePrefetchPending = prefetchPerLayerRow(
    tokenId,
    pleWeights.embedTokensPerLayer,
    config.numLayers * pleHiddenSize,
    resolvedPerLayerInputsSession
  );
}

export function getReusableSampleReadbackBuffer(state, device, size) {
  const existing = state.sampleReadbackBuffer;
  if (existing && existing.size >= size) {
    return existing;
  }
  if (existing) {
    existing.destroy();
  }
  const buffer = device.createBuffer({
    label: 'sample_staging_reuse',
    size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  state.sampleReadbackBuffer = buffer;
  return buffer;
}

export function shouldLogProfileStep(state, step) {
  const profilerConfig = state.runtimeConfig?.shared?.debug?.profiler;
  const every = profilerConfig?.logEveryDecodeSteps ?? 1;
  if (!Number.isFinite(every) || every <= 1) return true;
  return step === 1 || step % every === 0;
}

export function recordDecodeProfileStep(state, entry) {
  if (!entry || (!entry.timings && !entry.recorderStats)) return;
  if (!state.stats.decodeProfileSteps) {
    state.stats.decodeProfileSteps = [];
  }
  state.stats.decodeProfileSteps.push(entry);
}

export function shouldUseFusedDecodeSampling(config) {
  return config.recorderEnabled === true
    && config.gpuSamplingEnabled === true
    && config.fusedDecodeDisabled !== true
    && !hasConvLayers(config.layerTypes ?? []);
}

export async function runDecodeLayers(state, tokenId, opts, helpers) {
  const config = state.modelConfig;
  const debugCheckBuffer = state.debug ? helpers.debugCheckBuffer : undefined;

  const context = helpers.buildLayerContext(undefined, true, opts.debugLayers, opts.executionPlan);
  context.currentTokenIds = [tokenId];
  opts.__activationTrace?.activity?.begin('decode', state.currentSeqLen, [tokenId]);
  context.signal = opts.signal;

  state.decodeBuffers.resetPingPong();

  const decodeHiddenBuffer = state.decodeBuffers.getHiddenBuffer();
  const decodeAltBuffer = state.decodeBuffers.getOutputHiddenBuffer();

  const embedBufferRaw = state.weights.get('embed');
  if (!isGpuBufferInstance(embedBufferRaw) && !isWeightBuffer(embedBufferRaw) && !isCpuWeightBuffer(embedBufferRaw) && !isSplitWeightBuffer(embedBufferRaw) && !(embedBufferRaw instanceof Float32Array)) {
    throw new Error('Embed buffer not found or not a supported buffer type');
  }
  const embedBuffer = isWeightBuffer(embedBufferRaw) ? embedBufferRaw.buffer : embedBufferRaw;
  const embedDtype = isCpuWeightBuffer(embedBufferRaw)
    ? embedBufferRaw.dtype
    : getWeightDtype(embedBufferRaw);
  const embedMetadata = getWeightMetadata(embedBufferRaw);
  const activationDtype = getEffectiveActivationDtype(state, opts);
  const representation = getActivationRepresentation(state, opts.executionPlan);

  const embedTensor = await embed([tokenId], embedBuffer, {
    hiddenSize: config.hiddenSize,
    vocabSize: config.vocabSize,
    scaleEmbeddings: config.scaleEmbeddings,
    embeddingScale: config.embeddingScale, embeddingNormalization: config.embeddingNormalization,
    outputBuffer: decodeHiddenBuffer ?? undefined,
    transpose: state.embeddingTranspose,
    debugProbes: state.runtimeConfig.shared.debug.probes,
    operatorDiagnostics: state.operatorDiagnostics,
    activationDtype,
    intermediatePrecision: representation.precision,
    embeddingDtype: selectRuleValue('inference', 'dtype', 'embeddingDtype', { dtype: embedDtype }),
    embeddingStorageEncoding: embedMetadata?.storageEncoding ?? null,
    executionPolicies: state.executionV1State?.policies ?? null,
  });

  let hiddenStates = embedTensor.buffer;

  let returned = false;
  try {
    // Resolve pending PLE prefetch from previous decode step
    let plePrefetchResult = null;
    if (state.plePrefetchPending) {
      plePrefetchResult = await state.plePrefetchPending;
      state.plePrefetchPending = null;
    }

    const perLayerInputs = await preparePerLayerInputs([tokenId], embedTensor, context, {
      numTokens: 1,
      pleCache: state.pleCache ?? null,
      prefetchedRow: plePrefetchResult,
    });

    try {
      for (let l = 0; l < config.numLayers; l++) {
        context.perLayerInputBuffer = perLayerInputs?.[l] ?? null;
        const prevStates = hiddenStates;
        hiddenStates = (await processLayer(l, hiddenStates, 1, false, context));
        opts.__activationTrace?.capture(hiddenStates, { layer: l, phase: 'decode', tokenStart: state.currentSeqLen,
          tokenIds: [tokenId], dtype: activationDtype, ...representation });
        state.decodeBuffers.swapPingPong();
        releasePerLayerInputBuffer(context.perLayerInputBuffer, null, context.decodeBuffers, state.pleCache ?? null);
        if (perLayerInputs) {
          perLayerInputs[l] = null;
        }
        context.perLayerInputBuffer = null;
        if (isGpuBufferInstance(prevStates) && prevStates !== hiddenStates) {
          const isPreAllocated = isOwnedDecodeBuffer(prevStates, decodeHiddenBuffer, decodeAltBuffer);
          if (!isPreAllocated) {
            releaseBuffer(prevStates);
          }
        }
      }
    } finally {
      context.perLayerInputBuffer = null;
      if (perLayerInputs) {
        for (const buffer of perLayerInputs) {
          releasePerLayerInputBuffer(buffer, null, context.decodeBuffers, state.pleCache ?? null);
        }
      }
      helpers.releaseSharedAttentionState?.(context.sharedAttentionState, null);
    }

    returned = true;
    return { hiddenStates, decodeHiddenBuffer, decodeAltBuffer, debugCheckBuffer, context };
  } finally {
    if (!returned && !isOwnedDecodeBuffer(hiddenStates, decodeHiddenBuffer, decodeAltBuffer)) releaseBuffer(hiddenStates);
  }
}

export function createDecodeRecorder(state, opts) {
  const device = getDevice();
  const executionPlan = opts.executionPlan;
  const recorderConfig = {
    hasDevice: Boolean(device),
    debug: opts.debug,
    disableCommandBatching: executionPlan?.disableCommandBatching ?? opts.disableCommandBatching,
    kvLayout: state.kvCache?.layout ?? null,
  };
  const recorderEnabled = opts.profile
    ? isProfileDecodeRecorderEnabled(recorderConfig)
    : isDecodeRecorderEnabled(recorderConfig);
  let recorder;
  if (recorderEnabled) {
    const recorderOptions = {
      recordLabels: opts.debug === true || opts.benchmark === true || opts.executionObserver === true,
      recordDispatches: opts.debug === true || opts.executionObserver === true,
      aggregateDispatches: opts.executionObserver === true,
    };
    recorder = opts.profile
      ? createProfilingRecorder('decode', device, { ...recorderOptions, profilePhase: 'decode' })
      : createCommandRecorder('decode', recorderOptions, device);
  }
  if (state.decodeStepCount === 1) {
    const path = selectRuleValue('inference', 'config', 'tracePath', { useRecorder: Boolean(recorder) });
    log.debug('Decode', `Using ${path} path (recorder=${!!recorder}, debug=${opts.debug})`);
  }
  return recorder;
}

export async function submitDecodeRecorderProfile(state, opts, recorder, profileLabel) {
  if (!recorder) {
    return;
  }
  await recorder.submitAndWait();

  if (!opts.profile) {
    return;
  }

  const timings = await recorder.resolveProfileTimings({ requireComplete: true });
  const total = sumProfileTimings(timings);
  if (total !== null) {
    state.stats.gpuTimeDecodeMs = (state.stats.gpuTimeDecodeMs ?? 0) + total;
  }
  if (timings || opts.executionObserver) {
    recordDecodeProfileStep(state, {
      step: state.decodeStepCount,
      timings,
      totalMs: total ?? undefined,
      recorderStats: recorder.getStats(),
    });
    if (!opts.executionObserver && shouldLogProfileStep(state, state.decodeStepCount)) {
      log.warn('Profile', `Decode step ${state.decodeStepCount}${profileLabel}:`);
      log.warn('Profile', CommandRecorder.formatProfileReport(timings));
    }
  }
}

export function resolveDecodeLayerSubmitInterval(decodeLoop, recorder) {
  const interval = decodeLoop?.layerSubmitInterval ?? null;
  if (interval === null) return null;
  if (!Number.isSafeInteger(interval) || interval < 1) {
    throw new Error('decodeLoop.layerSubmitInterval must be a positive integer or null.');
  }
  if (!recorder) {
    throw new Error('decodeLoop.layerSubmitInterval requires a recorded decode step; the unrecorded decode path cannot use it.');
  }
  return interval;
}
