import type { ShaderSourceScope } from '../../gpu/kernels/shader-source-scope.js';
export type PipelineOperationKind = 'execution' | 'streaming' | 'mutation' | 'reset' | 'inspection' | 'shutdown';
export type PipelineOperationContract = Readonly<Record<string, PipelineOperationKind
  | Readonly<{ kind: 'execution'; context: 'explicit' }>>>;
export declare const PIPELINE_OPERATIONS: PipelineOperationContract;
/** One owner per pipeline; unload seals it immediately and drains queued cleanup. */
export declare function scopePipelineShaders<T extends object>(pipeline: T, scope?: ShaderSourceScope | null,
  operations?: PipelineOperationContract): T;
/** Retain the session through a host operation and its evidence construction. */
export declare function runPipelineOperation<P extends object, T>(pipeline: P, action: (pipeline: P) => T | Promise<T>): Promise<T>;
/** Prepare and activate an adapter exclusively; closing prevents late activation. */
export declare function updatePipelineAdapter<T>(
  pipeline: { setLoRAAdapter(adapter: T): void },
  prepare: () => T | Promise<T>
): Promise<T>;
