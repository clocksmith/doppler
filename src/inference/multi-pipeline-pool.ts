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

export class MultiPipelinePool {
  private loader: MultiModelLoader;
  private pipelines: Map<string, InferencePipeline>;
  private pipelineLocks: Map<string, Promise<void>>;
  private defaultContexts: PipelineContexts;
  private partitionedPool: PartitionedBufferPool | null;
  private recorder: MultiModelRecorder | null;
  private sharedPrefix: KVCacheSnapshot | null;

  constructor(loader: MultiModelLoader, options: MultiPipelinePoolOptions = {}) {
    this.loader = loader;
    this.pipelines = new Map();
    this.pipelineLocks = new Map();
    this.defaultContexts = options.contexts ?? {};
    this.partitionedPool = options.partitionConfig
      ? new PartitionedBufferPool(options.partitionConfig)
      : null;
    this.recorder = options.recorder ?? null;
    this.sharedPrefix = null;
  }

  setRecorder(recorder: MultiModelRecorder | null): void {
    this.recorder = recorder;
  }

  getRecorder(): MultiModelRecorder | null {
    return this.recorder;
  }

  getPartitionedPool(): PartitionedBufferPool | null {
    return this.partitionedPool;
  }

  setSharedPrefixSnapshot(snapshot: KVCacheSnapshot | null): void {
    this.sharedPrefix = snapshot;
    if (this.recorder) {
      this.recorder.setSharedPrefix(snapshot);
    }
  }

  getSharedPrefixSnapshot(): KVCacheSnapshot | null {
    return this.recorder?.getSharedPrefix() ?? this.sharedPrefix;
  }

  private mergeContexts(contexts?: PipelineContexts): PipelineContexts {
    if (!contexts) return { ...this.defaultContexts };
    return {
      ...this.defaultContexts,
      ...contexts,
      gpu: {
        ...this.defaultContexts.gpu,
        ...contexts.gpu,
      },
      storage: {
        ...this.defaultContexts.storage,
        ...contexts.storage,
      },
      runtime: {
        ...this.defaultContexts.runtime,
        ...contexts.runtime,
      },
    };
  }

  async getPipeline(
    id: string,
    contexts: PipelineContexts = {}
  ): Promise<InferencePipeline> {
    const existing = this.pipelines.get(id);
    if (existing) return existing;

    const pipeline = await this.loader.createSharedPipeline(this.mergeContexts(contexts));
    this.pipelines.set(id, pipeline);
    return pipeline;
  }

  listPipelines(): string[] {
    return Array.from(this.pipelines.keys());
  }

  async warmPool(ids: string[], contexts: PipelineContexts = {}): Promise<void> {
    await Promise.all(ids.map((id) => this.getPipeline(id, contexts)));
  }

  async unloadAll(): Promise<void> {
    const pipelines = Array.from(this.pipelines.values());
    await Promise.all(pipelines.map(async (pipeline) => pipeline.unload()));
    this.pipelines.clear();
    this.pipelineLocks.clear();
  }

  private async withPipelineLock<T>(id: string, fn: () => Promise<T>): Promise<T> {
    const previous = this.pipelineLocks.get(id) || Promise.resolve();
    let release: (() => void) | null = null;
    const current = new Promise<void>((resolve) => {
      release = resolve;
    });
    this.pipelineLocks.set(id, previous.then(() => current));
    await previous;
    try {
      return await fn();
    } finally {
      release?.();
      if (this.pipelineLocks.get(id) === current) {
        this.pipelineLocks.delete(id);
      }
    }
  }

  async execute(
    id: string,
    prompt: string,
    options: GenerateOptions = {},
    adapter?: LoRAAdapter | null,
    prefix?: KVCacheSnapshot | null
  ): Promise<string> {
    const resolvedPrefix = prefix ?? this.getSharedPrefixSnapshot();

    return this.withPipelineLock(id, async () => {
      const pipeline = await this.getPipeline(id);
      pipeline.setLoRAAdapter(adapter || null);

      const generator = resolvedPrefix
        ? pipeline.generateWithPrefixKV(resolvedPrefix, prompt, options)
        : pipeline.generate(prompt, options);

      const chunks: string[] = [];
      for await (const token of generator) {
        chunks.push(token);
      }
      return chunks.join('');
    });
  }
}
