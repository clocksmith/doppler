export { DOPPLER_VERSION } from './version.js';
export { createDopplerRuntime, RUNTIME_CORE_VERSION } from './client/runtime/composition-root.js';
export { createFetchPackArtifactStore } from './client/runtime/fetch-pack-artifact-store.js';

import { createDopplerRuntime } from './client/runtime/composition-root.js';

export function openPack(packOrId, options = {}) {
  const required = ['device', 'artifactStore', 'trustedSigners', 'programFactory'];
  const missing = required.filter((field) => options[field] == null);
  if (missing.length > 0) {
    throw new Error(`Pack runtime openPack() requires explicit ports: ${missing.join(', ')}.`);
  }
  const runtime = createDopplerRuntime({
    device: options.device,
    packSource: options.packSource ?? null,
    artifactStore: options.artifactStore,
    trustedSigners: options.trustedSigners,
    programFactory: options.programFactory,
    cache: options.verificationCache ?? null,
    observer: options.observer ?? null,
  });
  return runtime.openPack(packOrId, options.session ?? {});
}
