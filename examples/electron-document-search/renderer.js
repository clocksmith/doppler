import { createDopplerRuntime } from 'doppler-gpu';
import { createElectronRendererRuntime } from 'doppler-gpu/electron';
import { openCapsule } from 'doppler-gpu/host';

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
