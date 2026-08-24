export function destroyMoERouter(router) {
  if (router && typeof router.destroy === 'function') {
    router.destroy();
  }
}

function createPipelineLoadPhaseError(error, phase, context = {}) {
  const message = error?.message || String(error);
  const wrapped = new Error(
    `Pipeline load phase "${phase}" failed: ${message}`,
    error instanceof Error ? { cause: error } : undefined
  );
  wrapped.name = error?.name || 'Error';
  if (error?.code !== undefined) {
    wrapped.code = error.code;
  }
  wrapped.details = {
    ...(error?.details && typeof error.details === 'object' ? error.details : {}),
    pipelineLoadPhase: phase,
    ...context,
  };
  return wrapped;
}

async function withPipelineLoadPhase(phase, context, run) {
  try {
    return await run();
  } catch (error) {
    if (error?.details?.pipelineLoadPhase) {
      throw error;
    }
    throw createPipelineLoadPhaseError(error, phase, context);
  }
}

export function roundPipelineTimingMs(value) {
  return Number.isFinite(value) ? Number(value.toFixed(3)) : null;
}

export function createPipelineLoadTiming(modelId) {
  return {
    schemaVersion: 1,
    source: 'doppler-pipeline',
    modelId: typeof modelId === 'string' ? modelId : null,
    status: 'running',
    phasesMs: {
      reset: null,
      configResolution: null,
      kernelWarmup: null,
      tokenizer: null,
      executionSetup: null,
      loadWeights: null,
      rope: null,
      convStates: null,
    },
    details: { tokenizer: null },
    totalMs: null,
  };
}

export function finishPipelineLoadTimingPhase(loadTiming, phase, startMs) {
  if (!loadTiming?.phasesMs || !phase) return;
  const elapsedMs = roundPipelineTimingMs(performance.now() - startMs);
  const currentMs = loadTiming.phasesMs[phase];
  loadTiming.phasesMs[phase] = Number.isFinite(currentMs)
    ? roundPipelineTimingMs(currentMs + elapsedMs)
    : elapsedMs;
}

export async function timedPipelineLoadPhase(loadTiming, phase, context, run) {
  const startMs = performance.now();
  try {
    return await withPipelineLoadPhase(phase, context, run);
  } finally {
    finishPipelineLoadTimingPhase(loadTiming, phase, startMs);
  }
}
