export { DOPPLER_VERSION } from './version.js';
export { createDopplerRuntime, RUNTIME_CORE_VERSION } from './client/runtime/composition-root.js';
export { createForecastProgramFactory } from './inference/pipelines/forecast/pack-program.js';
export { createFetchPackArtifactStore } from './client/runtime/fetch-pack-artifact-store.js';

import type { DopplerPack } from './config/pack.js';
import type {
  DopplerRuntimeSession,
  RuntimePorts,
  PackSessionOptions,
} from './client/runtime/composition-root.js';

export type { DopplerRuntime, DopplerRuntimeSession, RuntimePorts } from './client/runtime/composition-root.js';
export type { PackRerankApplicationBinding, PackRerankRequest, PackRerankReceipt } from './client/runtime/pack-rerank.js';

export declare function openPack(
  packOrId: string | DopplerPack,
  options: Omit<RuntimePorts, 'packSource' | 'cache'> & {
    packSource?: RuntimePorts['packSource'];
    verificationCache?: RuntimePorts['cache'];
    session?: PackSessionOptions;
  }
): Promise<DopplerRuntimeSession>;
