import { ERROR_CODES } from '../../errors/doppler-error.js';
import { normalizeTargetPlanSelectionPolicy } from '../../config/target-plan.js';

function cancellationError() {
  const error = new Error('Electron renderer Capsule operation was cancelled.');
  error.name = 'AbortError';
  error.code = 'DOPPLER_ELECTRON_CANCELLED';
  return error;
}

function assertActive(signal) {
  if (signal?.aborted) throw cancellationError();
}

function deviceLossError(cause) {
  const error = new Error(`Electron renderer WebGPU device was lost: ${cause.message}`);
  error.name = 'DopplerElectronDeviceLostError';
  error.code = 'DOPPLER_ELECTRON_DEVICE_LOST';
  error.cause = cause;
  return error;
}

function assertSameCapsule(actual, expected) {
  if (actual?.capsuleId !== expected.capsuleId || actual?.semanticRoot !== expected.semanticRoot) {
    const error = new Error('Electron Capsule session does not match the current authorized release.');
    error.code = 'DOPPLER_ELECTRON_RELEASE_CHANGED';
    throw error;
  }
}

function translateError(error) {
  if (error?.code === ERROR_CODES.GPU_DEVICE_LOST || error?.code === 'GPU_DEVICE_LOST' || error?.name === 'GPUDeviceLostError') {
    return deviceLossError(error);
  }
  return error;
}

export function createElectronRendererRuntime(options) {
  if (typeof options?.releaseState?.resolveCurrent !== 'function') {
    throw new Error('Electron renderer runtime requires releaseState.resolveCurrent().');
  }
  if (typeof options.openCapsule !== 'function') {
    throw new Error('Electron renderer runtime requires openCapsule().');
  }

  async function openCurrent(openOptions = {}) {
    assertActive(openOptions.signal);
    const capsule = await options.releaseState.resolveCurrent();
    assertActive(openOptions.signal);
    let session;
    try {
      session = await options.openCapsule(capsule.path, openOptions);
      assertSameCapsule(session, capsule);
      assertActive(openOptions.signal);
      assertSameCapsule(await options.releaseState.resolveCurrent(), capsule);
      assertActive(openOptions.signal);
    } catch (error) {
      try {
        await session?.close();
      } catch {
        // Preserve the load, authorization, or cancellation failure.
      }
      assertActive(openOptions.signal);
      throw translateError(error);
    }
    return session;
  }

  async function rerank(request, openOptions = {}) {
    if (!request || typeof request !== 'object' || Array.isArray(request) || !request.application) {
      throw new Error('Electron rerank requires a CapsuleRerankRequest with an explicit application binding.');
    }
    const signals = [openOptions.signal, request.options?.signal].filter(Boolean);
    const signal = signals.length === 2 && signals[0] !== signals[1]
      ? AbortSignal.any(signals) : signals[0];
    const policy = normalizeTargetPlanSelectionPolicy({ requiredOperations: openOptions.requiredOperations });
    const requiredOperations = [...new Set([...(policy.requiredOperations ?? []), 'rerank'])];
    const session = await openCurrent({ ...openOptions, requiredOperations, ...(signal ? { signal } : {}) });
    let failed = false;
    try {
      if (typeof session.rerank !== 'function') {
        throw new Error('Electron current Capsule does not expose the qualified reranking workload.');
      }
      const result = await session.rerank(signal
        ? { ...request, options: { ...request.options, signal } } : request);
      assertActive(signal);
      assertSameCapsule(await options.releaseState.resolveCurrent(), session);
      assertActive(signal);
      return result;
    } catch (error) {
      failed = true;
      assertActive(signal);
      throw translateError(error);
    } finally {
      try {
        await session.close();
      } catch (error) {
        if (!failed) throw translateError(error);
      }
    }
  }

  return Object.freeze({ openCurrent, rerank });
}
