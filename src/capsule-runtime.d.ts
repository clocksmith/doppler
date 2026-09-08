export { DOPPLER_VERSION } from './version.js';
export { GENERATION_CONTRACT, GenerationError, resolveGenerationOptions, validateGenerationInput } from './config/generation-contract.js';
export type { GenerationInput, GenerationOptions, ResolvedGenerationOptions, GenerationOutput, GenerationCompletion } from './config/generation-contract.js';
export { createDopplerRuntime, createForecastProgramFactory, RUNTIME_CORE_VERSION } from './client/runtime/composition-root.js';
export { createFetchCapsuleArtifactStore } from './client/runtime/fetch-capsule-artifact-store.js';

import type { DopplerCapsule } from './config/capsule.js';
import type {
  DopplerRuntimeSession,
  RuntimePorts,
  CapsuleSessionOptions,
} from './client/runtime/composition-root.js';

export type { DopplerRuntime, DopplerRuntimeSession, RuntimePorts } from './client/runtime/composition-root.js';
export type { CapsuleRerankApplicationBinding, CapsuleRerankRequest, CapsuleRerankReceipt } from './client/runtime/capsule-rerank.js';
export type { CapsuleEmbeddingRequest, CapsuleEmbeddingResult } from './client/runtime/composition-root.js';

export declare function openCapsule(
  capsuleOrId: string | DopplerCapsule,
  options: Omit<RuntimePorts, 'capsuleSource' | 'cache'> & {
    capsuleSource?: RuntimePorts['capsuleSource'];
    verificationCache?: RuntimePorts['cache'];
    session?: CapsuleSessionOptions;
  }
): Promise<DopplerRuntimeSession>;
