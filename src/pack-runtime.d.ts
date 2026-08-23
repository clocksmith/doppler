export { DOPPLER_VERSION } from './version.js';
export { createDopplerRuntime, RUNTIME_CORE_VERSION } from './client/runtime/composition-root.js';
export { createFetchPackArtifactStore } from './client/runtime/fetch-pack-artifact-store.js';

import type { DopplerPackV2 } from './config/pack-v2.js';
import type {
  DopplerRuntimeSession,
  RuntimePorts,
} from './client/runtime/composition-root.js';

export declare function openPack(
  packOrId: string | DopplerPackV2,
  options: Omit<RuntimePorts, 'packSource' | 'cache'> & {
    packSource?: RuntimePorts['packSource'];
    verificationCache?: RuntimePorts['cache'];
    session?: Record<string, unknown>;
  }
): Promise<DopplerRuntimeSession>;
