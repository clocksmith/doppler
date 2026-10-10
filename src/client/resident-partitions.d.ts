import type { DopplerCapsule } from '../config/capsule.js';
import type { GenerationOptions } from '../config/generation-contract.js';
import type { DopplerCapsuleOpenOptions } from './model-host/index.js';
import type { DopplerRunSession } from './runtime/composition-root.js';
import type { ResidentPartitionAllocation, ResidentPartitionSession } from '../inference/pipelines/text/resident-partition-contract.js';
export function configureDeviceMemoryBudget(options: {maxBytes: number | null}): Promise<import('../memory/device-budget.js').DeviceMemorySnapshot>;
export function inspectDeviceMemory(): import('../memory/device-budget.js').DeviceMemorySnapshot | null;
export interface ResidentPartitionOpenOptions extends Omit<ResidentPartitionAllocation, 'generation' | 'model'> {
  model: { id: string; identity: string; capsule: string | DopplerCapsule; generation: GenerationOptions };
  signal: AbortSignal;
}
export interface ResidentPartitionFactory {
  /** The returned session declares recovery separately from readiness and signed model qualification. */
  openResidentPartition(options: ResidentPartitionOpenOptions): Promise<ResidentPartitionSession>;
}
export function createResidentPartitionFactory(options: {
  openCapsule(capsule: string | DopplerCapsule, options: DopplerCapsuleOpenOptions): Promise<DopplerRunSession>;
  capsuleOptions: DopplerCapsuleOpenOptions;
}): ResidentPartitionFactory;

/** Pinned RDRR development lane; does not assert signed Capsule qualification. */
export function createManifestResidentPartitionFactory(options: {
  manifest: import('../formats/rdrr/index.js').RDRRManifest; manifestIdentity: string;
  runtimeConfig: object;
  createStorage: (input: Omit<ResidentPartitionOpenOptions, 'model'> & {model: {id:string;identity:string;generation:GenerationOptions}}) => Promise<ReturnType<typeof import('../storage/artifact-storage-context.js').createArtifactStorageContext>>;
}): { openResidentPartition(options: Omit<ResidentPartitionOpenOptions, 'model'> & {model:{id:string;identity:string;generation:GenerationOptions}}): Promise<ResidentPartitionSession> };
