import type { PackV2Artifact } from '../../config/pack-v2.js';
import type { DopplerPack, PackIdentity } from '../../config/pack.js';
import type { PackReleaseEvent, PackReleasePolicy, ReleaseCheckpoint } from '../../config/pack-release-events.js';
import type { TargetPlan } from '../../config/target-plan.js';
import type { InitialExecutionIdentity } from '../../config/initial-execution-identity.js';
import type { DeviceProfile } from './target-selector.js';
import type { GenerationRunOptions } from './session-controller.js';
import type { PackRerankReceipt, PackRerankRequest } from './pack-rerank.js';
import type { PackOperationRequest } from '../../config/pack-operation.js';
import type { PackOperationEvent } from './pack-operation-executor.js';
import type { PackForecastRequest, PackForecastResult } from './pack-forecast.js';
import type { PackEmbeddingRequest, PackEmbeddingResult } from './pack-embedding.js';
export type { PackEmbeddingRequest, PackEmbeddingResult } from './pack-embedding.js';

export { createForecastProgramFactory } from './pack-forecast-program.js';

export const RUNTIME_CORE_VERSION: '2.0.0';

export interface PackSessionOptions {
  acceptedTargetPlanDigests?: string[];
  releaseEvents?: PackReleaseEvent[];
  releaseTrustedSigners?: Map<string, JsonWebKey> | Record<string, JsonWebKey>;
  releasePolicy?: PackReleasePolicy;
  persistReleaseCheckpoint?: (checkpoint: ReleaseCheckpoint) => Promise<void> | void;
}

export interface RuntimePorts {
  device: object;
  packSource?: { fetchPack(id: string, options?: object): Promise<DopplerPack> };
  artifactStore: {
    hashArtifact?(artifact: PackV2Artifact): Promise<{ hash: string; sizeBytes: number }>;
    readArtifact(artifact: PackV2Artifact): Promise<Uint8Array>;
  };
  trustedSigners: Map<string, JsonWebKey> | Record<string, JsonWebKey>;
  programFactory(args: Record<string, unknown>): Promise<object>;
  cache?: { set(key: string, value: unknown): Promise<void> | void } | null;
  observer?: { observe(event: Record<string, unknown>): void } | null;
}

export interface DopplerRuntimeSession {
  schema: 'doppler.pack-session/v1';
  readonly loaded: boolean;
  readonly closed: boolean;
  packIdentity: PackIdentity;
  readonly manifest: Readonly<Record<string, unknown>>;
  readonly manifestHash: string;
  modelId: string;
  packId: string;
  semanticRoot: string;
  selectedTargetId: string;
  selectedTargetPlanDigest: string;
  selectedPlan: TargetPlan;
  observedInitialExecutionIdentity: InitialExecutionIdentity | null;
  deviceProfile: DeviceProfile;
  verification: {
    pack: DopplerPack;
    identity: PackIdentity;
    artifactReceipts: Array<Record<string, unknown>>;
  };
  generate(options: GenerationRunOptions): AsyncGenerator<number, void, void>;
  generateText(options: GenerationRunOptions): Promise<{ text: string; tokenIds: number[] }>;
  rerank(request: PackRerankRequest): Promise<PackRerankReceipt>;
  forecast(request: PackForecastRequest): Promise<PackForecastResult>;
  embed(request: PackEmbeddingRequest): Promise<PackEmbeddingResult>;
  encodeSequence(sequence: string, options?: Record<string, unknown> & { signal?: AbortSignal }): Promise<Record<string, unknown>>;
  executeOperation(request: PackOperationRequest, control?: { signal?: AbortSignal | null }): AsyncGenerator<PackOperationEvent, void, void>;
  resetGenerationState(): void;
  close(): Promise<void>;
}

export interface DopplerRuntime {
  version: string;
  ports: RuntimePorts;
  openPack(packOrId: string | DopplerPack, options?: PackSessionOptions): Promise<DopplerRuntimeSession>;
}

export declare function createDopplerRuntime(ports: RuntimePorts): DopplerRuntime;
