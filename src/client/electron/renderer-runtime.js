import { ERROR_CODES } from '../../errors/doppler-error.js';

function cancellationError() {
  const error = new Error('Electron renderer Pack operation was cancelled.');
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

function assertSamePack(actual, expected) {
  if (actual?.packId !== expected.packId || actual?.semanticRoot !== expected.semanticRoot) {
    const error = new Error('Electron Pack session does not match the current authorized release.');
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
  if (typeof options.openPack !== 'function') {
    throw new Error('Electron renderer runtime requires openPack().');
  }

  async function openCurrent(openOptions = {}) {
    assertActive(openOptions.signal);
    const pack = await options.releaseState.resolveCurrent();
    assertActive(openOptions.signal);
    let session;
    try {
      session = await options.openPack(pack.path, openOptions);
      assertSamePack(session, pack);
      assertActive(openOptions.signal);
      assertSamePack(await options.releaseState.resolveCurrent(), pack);
      assertActive(openOptions.signal);
    } catch (error) {
      try {
        await session?.close();
      } catch {
        // Preserve the load, authorization, or cancellation failure.
      }
      throw translateError(error);
    }
    return session;
  }

  async function rerank(request, openOptions = {}) {
    if (!request || typeof request !== 'object' || Array.isArray(request) || !request.application) {
      throw new Error('Electron rerank requires a PackRerankRequest with an explicit application binding.');
    }
    const session = await openCurrent(openOptions);
    let failed = false;
    try {
      if (typeof session.rerank !== 'function') {
        throw new Error('Electron current Pack does not expose the qualified reranking workload.');
      }
      const result = await session.rerank(request);
      assertActive(openOptions.signal);
      assertSamePack(await options.releaseState.resolveCurrent(), session);
      assertActive(openOptions.signal);
      return result;
    } catch (error) {
      failed = true;
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
