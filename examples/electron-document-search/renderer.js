import { openPack } from 'doppler-gpu';
import { createElectronRendererRuntime } from 'doppler-gpu/electron';

export function createDocumentSearchRenderer(releaseState) {
  return createElectronRendererRuntime({ releaseState, openPack });
}
