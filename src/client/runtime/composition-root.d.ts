import type { CapsuleAdapterArtifactStore } from './capsule-adapter-execution.js';
import type { CapsuleV2Artifact } from '../../config/capsule-v2.js';
import type { DopplerCapsule, CapsuleIdentity, verifyCapsule } from '../../config/capsule.js';
import type { CapsuleReleaseEvent, CapsuleReleasePolicy, ReleaseCheckpoint } from '../../config/capsule-release-events.js';
import type { TargetPlan, TargetPlanSelectionPolicy } from '../../config/target-plan.js';
import type { InitialExecutionIdentity } from '../../config/initial-execution-identity.js';
import type { DeviceProfile } from './target-selector.js';
import type { GenerationRunOptions } from './session-controller.js';
import type { CapsuleRerankReceipt, CapsuleRerankRequest } from './capsule-rerank.js';
import type { CapsuleOperationRequest } from '../../config/capsule-operation.js';
import type { CapsuleOperationEvent } from './capsule-operation-executor.js';
import type { CapsuleForecastRequest, CapsuleForecastResult } from './capsule-forecast.js';
import type { CapsuleEmbeddingRequest, CapsuleEmbeddingResult } from './capsule-embedding.js';
import type { CapsuleAcquisitionOptions } from './capsule-acquisition.js';
export type { CapsuleEmbeddingRequest, CapsuleEmbeddingResult } from './capsule-embedding.js';

export { createForecastProgramFactory } from './capsule-forecast-program.js';

export const RUNTIME_CORE_VERSION: '2.0.0';

export interface CapsuleSessionOptions extends TargetPlanSelectionPolicy, CapsuleAcquisitionOptions {
  releaseEvents?: CapsuleReleaseEvent[];
  releaseTrustedSigners?: Map<string, JsonWebKey> | Record<string, JsonWebKey>;
  releasePolicy?: CapsuleReleasePolicy;
  persistReleaseCheckpoint?: (checkpoint: ReleaseCheckpoint) => Promise<void> | void;
}

export interface RuntimePorts {
  device: object;
  capsuleSource?: { fetchCapsule(id: string, options?: object): Promise<DopplerCapsule> };
  artifactStore: {
    hashArtifact?(artifact: CapsuleV2Artifact): Promise<{ hash: string; sizeBytes: number }>;
    readArtifact(artifact: CapsuleV2Artifact, options?: CapsuleAcquisitionOptions): Promise<Uint8Array | ArrayBuffer>;
  };
  trustedSigners: Map<string, JsonWebKey> | Record<string, JsonWebKey>;
  programFactory(args: Record<string, unknown>): Promise<object>;
  cache?: { set(key: string, value: unknown): Promise<void> | void } | null;
  observer?: { observe(event: Record<string, unknown>): void } | null;
}

export interface DopplerRuntimeSession {
  schema: 'doppler.capsule-session/v1';
  readonly loaded: boolean;
  readonly closed: boolean;
  capsuleIdentity: CapsuleIdentity;
  readonly manifest: Readonly<Record<string, unknown>>;
  readonly manifestHash: string;
  modelId: string;
  capsuleId: string;
  semanticRoot: string;
  selectedTargetId: string;
  selectedTargetPlanDigest: string;
  selectedPlan: TargetPlan;
  observedInitialExecutionIdentity: InitialExecutionIdentity | null;
  deviceProfile: DeviceProfile;
  verification: Awaited<ReturnType<typeof verifyCapsule>>;
  generate(options: GenerationRunOptions): AsyncGenerator<number, void, void>;
  generateText(options: GenerationRunOptions): Promise<{ text: string; tokenIds: number[] }>;
  rerank(request: CapsuleRerankRequest): Promise<CapsuleRerankReceipt>;
  forecast(request: CapsuleForecastRequest): Promise<CapsuleForecastResult>;
  embed(request: CapsuleEmbeddingRequest): Promise<CapsuleEmbeddingResult>;
  encodeSequence(sequence: string, options?: Record<string, unknown> & { signal?: AbortSignal }): Promise<Record<string, unknown>>;
  executeOperation(request: CapsuleOperationRequest, control?: { signal?: AbortSignal | null; adapterArtifactStore?: CapsuleAdapterArtifactStore | null }): AsyncGenerator<CapsuleOperationEvent, void, void>;
  resetGenerationState(): void | Promise<void>;
  close(): Promise<void>;
}

export interface DopplerRuntime {
  version: string;
  ports: RuntimePorts;
  openCapsule(capsuleOrId: string | DopplerCapsule, options?: CapsuleSessionOptions): Promise<DopplerRuntimeSession>;
}

export declare function createDopplerRuntime(ports: RuntimePorts): DopplerRuntime;
