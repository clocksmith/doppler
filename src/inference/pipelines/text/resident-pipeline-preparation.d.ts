import type { KernelPipelineIdentity, KernelPipelineRequest } from '../../../gpu/kernels/pipeline-cache.js';
import type { PartitionExecutionState } from './partition-execution.js';
import type { ResidentLazyPipelineScope, ResidentPartitionAllocation, ResidentPipelinePreparation,
  ResidentPipelinePreparationProgress } from './resident-partition-contract.js';

export declare const RESIDENT_PIPELINE_PREPARATION_SCHEMA: 'doppler.resident-pipeline-preparation/v1';

/** One compiled program a stage can request, with the program roles that request it. */
export interface ResidentPipelineEntry {
  /** `shaderFile#entryPoint` plus canonical resolved constants. */
  key: string;
  request: KernelPipelineRequest;
  identity: KernelPipelineIdentity;
  roles: string[];
}

export interface ResidentPipelineProgram {
  pipelines: ResidentPipelineEntry[];
  lazy: ResidentLazyPipelineScope[];
}

/**
 * Enumerate, without GPU work, every compute pipeline the loaded stage can
 * request during prefill (every chunk) and decode, from its resolved execution
 * plan, kernel path, rules and loaded weight encodings.
 */
export declare function resolveResidentPipelineProgram(owner: PartitionExecutionState,
  allocation: ResidentPartitionAllocation): ResidentPipelineProgram;

/**
 * Create every enumerated pipeline on the stage device, one at a time, before
 * readiness, within `allocation.limits.maxPreparationMs`. Progress is reported
 * at each pipeline and on every status change. Abort or the bound starts no
 * further creation and is reported at once; a started creation stays owned and
 * the promise settles only after it does. Failure, timeout (code
 * DOPPLER_RESIDENT_PREPARATION_TIMEOUT) and cancellation reject naming the pipeline.
 */
export declare function prepareResidentPartitionPipelines(owner: PartitionExecutionState,
  allocation: ResidentPartitionAllocation, signal?: AbortSignal,
  onProgress?: ((progress: ResidentPipelinePreparationProgress) => void) | null): Promise<ResidentPipelinePreparation>;
