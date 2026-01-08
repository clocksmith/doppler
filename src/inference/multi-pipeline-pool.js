/**
 * Multi-pipeline pool for parallel expert execution.
 *
 * @module inference/multi-pipeline-pool
 */

import { PartitionedBufferPool } from '../gpu/partitioned-buffer-pool.js';
import { MultiModelRecorder } from '../gpu/multi-model-recorder.js';

/**
 * @typedef {import('./pipeline.js').InferencePipeline} InferencePipeline
 * @typedef {import('./pipeline.js').KVCacheSnapshot} KVCacheSnapshot
 * @typedef {import('./pipeline.js').GenerateOptions} GenerateOptions
 * @typedef {import('./pipeline.js').PipelineContexts} PipelineContexts
 * @typedef {import('./pipeline/lora.js').LoRAAdapter} LoRAAdapter
 * @typedef {import('../loader/multi-model-loader.js').MultiModelLoader} MultiModelLoader
 * @typedef {import('../gpu/partitioned-buffer-pool.js').PartitionConfig} PartitionConfig
 * @typedef {import('./multi-pipeline-pool.js').MultiPipelinePoolOptions} MultiPipelinePoolOptions
 */

export class MultiPipelinePool {
  /** @type {MultiModelLoader} */
  loader;

  /** @type {Map<string, InferencePipeline>} */
  pipelines;

  /** @type {Map<string, Promise<void>>} */
  pipelineLocks;

  /** @type {PipelineContexts} */
  defaultContexts;

  /** @type {PartitionedBufferPool | null} */
  partitionedPool;

  /** @type {MultiModelRecorder | null} */
  recorder;

  /** @type {KVCacheSnapshot | null} */
  sharedPrefix;

  /**
   * @param {MultiModelLoader} loader
   * @param {MultiPipelinePoolOptions} [options={}]
   */
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

  /**
   * @param {MultiModelRecorder | null} recorder
   * @returns {void}
   */
  setRecorder(recorder) {
    this.recorder = recorder;
  }

  /**
   * @returns {MultiModelRecorder | null}
   */
  getRecorder() {
    return this.recorder;
  }

  /**
   * @returns {PartitionedBufferPool | null}
   */
  getPartitionedPool() {
    return this.partitionedPool;
  }

  /**
   * @param {KVCacheSnapshot | null} snapshot
   * @returns {void}
   */
  setSharedPrefixSnapshot(snapshot) {
    this.sharedPrefix = snapshot;
    if (this.recorder) {
      this.recorder.setSharedPrefix(snapshot);
    }
  }

  /**
   * @returns {KVCacheSnapshot | null}
   */
  getSharedPrefixSnapshot() {
    return this.recorder?.getSharedPrefix() ?? this.sharedPrefix;
  }

  /**
   * @param {PipelineContexts} [contexts]
   * @returns {PipelineContexts}
   * @private
   */
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

  /**
   * @param {string} id
   * @param {PipelineContexts} [contexts={}]
   * @returns {Promise<InferencePipeline>}
   */
  async getPipeline(id, contexts = {}) {
    const existing = this.pipelines.get(id);
    if (existing) return existing;

    const pipeline = await this.loader.createSharedPipeline(this.mergeContexts(contexts));
    this.pipelines.set(id, pipeline);
    return pipeline;
  }

  /**
   * @returns {string[]}
   */
  listPipelines() {
    return Array.from(this.pipelines.keys());
  }

  /**
   * @param {string[]} ids
   * @param {PipelineContexts} [contexts={}]
   * @returns {Promise<void>}
   */
  async warmPool(ids, contexts = {}) {
    await Promise.all(ids.map((id) => this.getPipeline(id, contexts)));
  }

  /**
   * @returns {Promise<void>}
   */
  async unloadAll() {
    const pipelines = Array.from(this.pipelines.values());
    await Promise.all(pipelines.map(async (pipeline) => pipeline.unload()));
    this.pipelines.clear();
    this.pipelineLocks.clear();
  }

  /**
   * @template T
   * @param {string} id
   * @param {() => Promise<T>} fn
   * @returns {Promise<T>}
   * @private
   */
  async withPipelineLock(id, fn) {
    const previous = this.pipelineLocks.get(id) || Promise.resolve();
    /** @type {(() => void) | null} */
    let release = null;
    const current = new Promise((resolve) => {
      release = /** @type {() => void} */ (resolve);
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

  /**
   * @param {string} id
   * @param {string} prompt
   * @param {GenerateOptions} [options={}]
   * @param {LoRAAdapter | null} [adapter]
   * @param {KVCacheSnapshot | null} [prefix]
   * @returns {Promise<string>}
   */
  async execute(id, prompt, options = {}, adapter, prefix) {
    const resolvedPrefix = prefix ?? this.getSharedPrefixSnapshot();

    return this.withPipelineLock(id, async () => {
      const pipeline = await this.getPipeline(id);
      pipeline.setLoRAAdapter(adapter || null);

      const generator = resolvedPrefix
        ? pipeline.generateWithPrefixKV(resolvedPrefix, prompt, options)
        : pipeline.generate(prompt, options);

      /** @type {string[]} */
      const chunks = [];
      for await (const token of generator) {
        chunks.push(token);
      }
      return chunks.join('');
    });
  }
}
