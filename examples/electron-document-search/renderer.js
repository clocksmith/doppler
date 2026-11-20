import { createDopplerRuntime } from 'dopplerx';
import { createElectronRendererRuntime } from 'dopplerx/host/electron';
import { openCapsule } from 'dopplerx/host';

export function createDocumentSearchHostRenderer(releaseState, trustOptions) {
  return createElectronRendererRuntime({
    releaseState,
    openCapsule: (capsulePath, options) => openCapsule(capsulePath, { ...trustOptions, ...options }),
  });
}

export function createDocumentSearchRenderer(releaseState, runtimePorts) {
  const runtime = createDopplerRuntime(runtimePorts);
  if (typeof runtimePorts.capsuleSource?.fetchCapsule !== 'function') {
    throw new Error('Document search requires capsuleSource.fetchCapsule() for authorized Capsule paths.');
  }
  return createElectronRendererRuntime({
    releaseState,
    openCapsule: (capsulePath, options) => runtime.openCapsule(capsulePath, options),
  });
}
