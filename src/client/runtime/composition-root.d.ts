import type { DopplerPackV2, PackV2Artifact } from '../../config/pack-v2.js';
import type { TargetPlan } from '../../config/target-plan.js';
import type { InitialExecutionIdentity } from '../../config/initial-execution-identity.js';
import type { DeviceProfile } from './target-selector.js';
import type { GenerationRunOptions } from './session-controller.js';

export const RUNTIME_CORE_VERSION: '2.0.0';

export interface RuntimePorts {
  device: object;
  packSource?: { fetchPack(id: string, options?: object): Promise<DopplerPackV2> };
  artifactStore: {
    hashArtifact(artifact: PackV2Artifact): Promise<{ hash: string; sizeBytes: number }>;
    readArtifact?(artifact: PackV2Artifact): Promise<Uint8Array>;
  };
  trustedSigners: Map<string, JsonWebKey> | Record<string, JsonWebKey>;
  programFactory(args: Record<string, unknown>): Promise<object>;
  cache?: { set(key: string, value: unknown): Promise<void> | void } | null;
  observer?: { observe(event: Record<string, unknown>): void } | null;
}

export interface DopplerRuntimeSession {
  modelId: string;
  packId: string;
  semanticRoot: string;
  selectedTargetId: string;
  selectedTargetPlanDigest: string;
  selectedPlan: TargetPlan;
  observedInitialExecutionIdentity: InitialExecutionIdentity | null;
  deviceProfile: DeviceProfile;
  verification: {
    pack: DopplerPackV2;
    artifactReceipts: Array<Record<string, unknown>>;
  };
  generate(options: GenerationRunOptions): AsyncGenerator<number, void, void>;
  generateText(options: GenerationRunOptions): Promise<{ text: string; tokenIds: number[] }>;
  close(): Promise<void>;
}

export interface DopplerRuntime {
  version: string;
  ports: RuntimePorts;
  openPack(packOrId: string | DopplerPackV2, options?: Record<string, unknown>): Promise<DopplerRuntimeSession>;
}

export declare function createDopplerRuntime(ports: RuntimePorts): DopplerRuntime;
