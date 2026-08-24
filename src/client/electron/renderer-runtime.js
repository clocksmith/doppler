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
    return options.openPack(pack.path, openOptions);
  }

  async function rerank(query, documents, runOptions = {}) {
    assertActive(runOptions.signal);
    const session = await openCurrent(runOptions);
    try {
      if (typeof session.rerank !== 'function') {
        throw new Error('Electron current Pack does not expose the qualified reranking workload.');
      }
      const result = await session.rerank(query, documents, runOptions);
      assertActive(runOptions.signal);
      return result;
    } catch (error) {
      if (error?.code === 'GPU_DEVICE_LOST' || error?.name === 'GPUDeviceLostError') {
        throw deviceLossError(error);
      }
      throw error;
    } finally {
      await session.close?.();
    }
  }

  return Object.freeze({ openCurrent, rerank });
}
