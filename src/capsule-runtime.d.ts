export { DOPPLER_VERSION } from './version.js';
export { GENERATION_CONTRACT, GenerationError, resolveGenerationOptions, validateGenerationInput } from './config/generation-contract.js';
export type { GenerationInput, GenerationOptions, ResolvedGenerationOptions, GenerationOutput, GenerationCompletion } from './config/generation-contract.js';
export { createDopplerRun, createDopplerRuntime, createForecastProgramFactory, RUN_CORE_VERSION, RUNTIME_CORE_VERSION } from './client/runtime/composition-root.js';
export { createFetchCapsuleArtifactStore } from './client/runtime/fetch-capsule-artifact-store.js';
export { createCapsuleStreamAccumulator, capsuleOperationSnapshots } from './client/runtime/capsule-operation-stream.js';
export type { CapsuleOperationRequest } from './config/capsule-operation.js';
export type { CapsuleOperationEvent } from './client/runtime/capsule-operation-executor.js';

import type { DopplerCapsule } from './config/capsule.js';
import type {
  DopplerRunSession,
  RunPorts,
  CapsuleSessionOptions,
} from './client/runtime/composition-root.js';

export type { DopplerRun, DopplerRunSession, RunPorts, DopplerRuntime, DopplerRuntimeSession, RuntimePorts } from './client/runtime/composition-root.js';
export type { CapsuleRerankApplicationBinding, CapsuleRerankRequest, CapsuleRerankReceipt } from './client/runtime/capsule-rerank.js';
export type { CapsuleEmbeddingRequest, CapsuleEmbeddingResult } from './client/runtime/composition-root.js';

export declare function openCapsule(
  capsuleOrId: string | DopplerCapsule,
  options: Omit<RunPorts, 'capsuleSource' | 'cache'> & {
    capsuleSource?: RunPorts['capsuleSource'];
    verificationCache?: RunPorts['cache'];
    session?: CapsuleSessionOptions;
  }
): Promise<DopplerRunSession>;
