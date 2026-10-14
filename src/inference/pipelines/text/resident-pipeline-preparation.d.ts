import type { KernelPipelineIdentity, KernelPipelineRequest } from '../../../gpu/kernels/pipeline-cache.js';
import type { PartitionExecutionState } from './partition-execution.js';
import type { ResidentLazyPipelineScope, ResidentPartitionAllocation, ResidentPipelinePreparation } from './resident-partition-contract.js';

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
 * readiness. Abort prevents later creations and waits for a started one to
 * settle. A creation failure rejects with an error naming the pipeline.
 */
export declare function prepareResidentPartitionPipelines(owner: PartitionExecutionState,
  allocation: ResidentPartitionAllocation, signal?: AbortSignal): Promise<ResidentPipelinePreparation>;
