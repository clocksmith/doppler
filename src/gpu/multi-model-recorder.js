

export class MultiModelRecorder {
  
  #sharedPrefix = null;

  
  async computeSharedPrefix(
    pipeline,
    prompt,
    options = {}
  ) {
    this.#sharedPrefix = await pipeline.prefillKVOnly(prompt, options);
    return this.#sharedPrefix;
  }

  
  getSharedPrefix() {
    return this.#sharedPrefix;
  }

  
  setSharedPrefix(snapshot) {
    this.#sharedPrefix = snapshot;
  }

  
  clear() {
    this.#sharedPrefix = null;
  }
}
