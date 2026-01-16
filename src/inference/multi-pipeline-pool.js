

import { PartitionedBufferPool } from '../gpu/partitioned-buffer-pool.js';
import { MultiModelRecorder } from '../gpu/multi-model-recorder.js';



export class MultiPipelinePool {
  
  loader;

  
  pipelines;

  
  pipelineLocks;

  
  defaultContexts;

  
  partitionedPool;

  
  recorder;

  
  sharedPrefix;

  
  constructor(loader, options = {}) {
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

  
  setRecorder(recorder) {
    this.recorder = recorder;
  }

  
  getRecorder() {
    return this.recorder;
  }

  
  getPartitionedPool() {
    return this.partitionedPool;
  }

  
  setSharedPrefixSnapshot(snapshot) {
    this.sharedPrefix = snapshot;
    if (this.recorder) {
      this.recorder.setSharedPrefix(snapshot);
    }
  }

  
  getSharedPrefixSnapshot() {
    return this.recorder?.getSharedPrefix() ?? this.sharedPrefix;
  }

  
  mergeContexts(contexts) {
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
      runtimeConfig: contexts.runtimeConfig ?? this.defaultContexts.runtimeConfig,
    };
  }

  
  async getPipeline(id, contexts = {}) {
    const existing = this.pipelines.get(id);
    if (existing) return existing;

    const pipeline = await this.loader.createSharedPipeline(this.mergeContexts(contexts));
    this.pipelines.set(id, pipeline);
    return pipeline;
  }

  
  listPipelines() {
    return Array.from(this.pipelines.keys());
  }

  
  async warmPool(ids, contexts = {}) {
    await Promise.all(ids.map((id) => this.getPipeline(id, contexts)));
  }

  
  async unloadAll() {
    const pipelines = Array.from(this.pipelines.values());
    await Promise.all(pipelines.map(async (pipeline) => pipeline.unload()));
    this.pipelines.clear();
    this.pipelineLocks.clear();
  }

  
  async withPipelineLock(id, fn) {
    const previous = this.pipelineLocks.get(id) || Promise.resolve();
    
    let release = null;
    const current = new Promise((resolve) => {
      release =  (resolve);
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

  
  async execute(id, prompt, options = {}, adapter, prefix) {
    const resolvedPrefix = prefix ?? this.getSharedPrefixSnapshot();

    return this.withPipelineLock(id, async () => {
      const pipeline = await this.getPipeline(id);
      pipeline.setLoRAAdapter(adapter || null);

      const generator = resolvedPrefix
        ? pipeline.generateWithPrefixKV(resolvedPrefix, prompt, options)
        : pipeline.generate(prompt, options);

      
      const chunks = [];
      for await (const token of generator) {
        chunks.push(token);
      }
      return chunks.join('');
    });
  }
}
