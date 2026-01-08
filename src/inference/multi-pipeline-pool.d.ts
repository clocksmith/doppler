/**
 * Multi-pipeline pool for parallel expert execution.
 *
 * @module inference/multi-pipeline-pool
 */

import type { InferencePipeline, KVCacheSnapshot, GenerateOptions, PipelineContexts } from './pipeline.js';
import type { LoRAAdapter } from './pipeline/lora.js';
import type { MultiModelLoader } from '../loader/multi-model-loader.js';
import { PartitionedBufferPool, type PartitionConfig } from '../gpu/partitioned-buffer-pool.js';
import { MultiModelRecorder } from '../gpu/multi-model-recorder.js';

export interface MultiPipelinePoolOptions {
  contexts?: PipelineContexts;
  partitionConfig?: PartitionConfig[];
  recorder?: MultiModelRecorder | null;
}

export declare class MultiPipelinePool {
  private loader;
  private pipelines;
  private pipelineLocks;
  private defaultContexts;
  private partitionedPool;
  private recorder;
  private sharedPrefix;

  constructor(loader: MultiModelLoader, options?: MultiPipelinePoolOptions);

  setRecorder(recorder: MultiModelRecorder | null): void;

  getRecorder(): MultiModelRecorder | null;

  getPartitionedPool(): PartitionedBufferPool | null;

  setSharedPrefixSnapshot(snapshot: KVCacheSnapshot | null): void;

  getSharedPrefixSnapshot(): KVCacheSnapshot | null;

  private mergeContexts(contexts?: PipelineContexts): PipelineContexts;

  getPipeline(
    id: string,
    contexts?: PipelineContexts
  ): Promise<InferencePipeline>;

  listPipelines(): string[];

  warmPool(ids: string[], contexts?: PipelineContexts): Promise<void>;

  unloadAll(): Promise<void>;

  private withPipelineLock<T>(id: string, fn: () => Promise<T>): Promise<T>;

  execute(
    id: string,
    prompt: string,
    options?: GenerateOptions,
    adapter?: LoRAAdapter | null,
    prefix?: KVCacheSnapshot | null
  ): Promise<string>;
}
