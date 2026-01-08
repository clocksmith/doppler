/**
 * Multi-model recorder for shared prefix KV caching.
 *
 * @module gpu/multi-model-recorder
 */

export class MultiModelRecorder {
  /** @type {import('../inference/pipeline.js').KVCacheSnapshot | null} */
  #sharedPrefix = null;

  /**
   * @param {import('../inference/pipeline.js').InferencePipeline} pipeline
   * @param {string} prompt
   * @param {import('../inference/pipeline.js').GenerateOptions} [options]
   * @returns {Promise<import('../inference/pipeline.js').KVCacheSnapshot>}
   */
  async computeSharedPrefix(
    pipeline,
    prompt,
    options = {}
  ) {
    this.#sharedPrefix = await pipeline.prefillKVOnly(prompt, options);
    return this.#sharedPrefix;
  }

  /**
   * @returns {import('../inference/pipeline.js').KVCacheSnapshot | null}
   */
  getSharedPrefix() {
    return this.#sharedPrefix;
  }

  /**
   * @param {import('../inference/pipeline.js').KVCacheSnapshot | null} snapshot
   * @returns {void}
   */
  setSharedPrefix(snapshot) {
    this.#sharedPrefix = snapshot;
  }

  /**
   * @returns {void}
   */
  clear() {
    this.#sharedPrefix = null;
  }
}
