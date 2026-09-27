import type { DopplerCapsule } from '../config/capsule.js';
import type { GenerationOptions } from '../config/generation-contract.js';
import type { DopplerCapsuleOpenOptions } from './model-host/index.js';
import type { DopplerRunSession } from './runtime/composition-root.js';
import type { ResidentPartitionAllocation, ResidentPartitionSession } from '../inference/pipelines/text/resident-partition-contract.js';
export interface ResidentPartitionOpenOptions extends Omit<ResidentPartitionAllocation, 'generation' | 'model'> {
  model: { id: string; identity: string; capsule: string | DopplerCapsule; generation: GenerationOptions };
  signal: AbortSignal;
}
export interface ResidentPartitionFactory {
  openResidentPartition(options: ResidentPartitionOpenOptions): Promise<ResidentPartitionSession>;
}
export function createResidentPartitionFactory(options: {
  openCapsule(capsule: string | DopplerCapsule, options: DopplerCapsuleOpenOptions): Promise<DopplerRunSession>;
  capsuleOptions: DopplerCapsuleOpenOptions;
}): ResidentPartitionFactory;
