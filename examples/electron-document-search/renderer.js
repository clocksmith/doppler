import { createDopplerRuntime } from 'doppler-gpu';
import { createElectronRendererRuntime } from 'doppler-gpu/electron';
import { openPack } from 'doppler-gpu/host';

export function createDocumentSearchHostRenderer(releaseState, trustOptions) {
  return createElectronRendererRuntime({
    releaseState,
    openPack: (packPath, options) => openPack(packPath, { ...trustOptions, ...options }),
  });
}

export function createDocumentSearchRenderer(releaseState, runtimePorts) {
  const runtime = createDopplerRuntime(runtimePorts);
  if (typeof runtimePorts.packSource?.fetchPack !== 'function') {
    throw new Error('Document search requires packSource.fetchPack() for authorized Pack paths.');
  }
  return createElectronRendererRuntime({
    releaseState,
    openPack: (packPath, options) => runtime.openPack(packPath, options),
  });
}
