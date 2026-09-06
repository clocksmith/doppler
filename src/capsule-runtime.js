export { DOPPLER_VERSION } from './version.js';
export { createDopplerRuntime, createForecastProgramFactory, RUNTIME_CORE_VERSION } from './client/runtime/composition-root.js';
export { createFetchCapsuleArtifactStore } from './client/runtime/fetch-capsule-artifact-store.js';

import { createDopplerRuntime } from './client/runtime/composition-root.js';

export function openCapsule(capsuleOrId, options = {}) {
  const required = ['device', 'artifactStore', 'trustedSigners', 'programFactory'];
  const missing = required.filter((field) => options[field] == null);
  if (missing.length > 0) {
    throw new Error(`Capsule runtime openCapsule() requires explicit ports: ${missing.join(', ')}.`);
  }
  const runtime = createDopplerRuntime({
    device: options.device,
    capsuleSource: options.capsuleSource ?? null,
    artifactStore: options.artifactStore,
    trustedSigners: options.trustedSigners,
    programFactory: options.programFactory,
    cache: options.verificationCache ?? null,
    observer: options.observer ?? null,
  });
  return runtime.openCapsule(capsuleOrId, options.session ?? {});
}
