/**
 * Multi-model recorder for shared prefix KV caching.
 *
 * @module gpu/multi-model-recorder
 */

import type { InferencePipeline, KVCacheSnapshot } from '../inference/pipeline.js';
import type { GenerateOptions } from '../inference/pipeline.js';

export class MultiModelRecorder {
  private sharedPrefix: KVCacheSnapshot | null = null;

  async computeSharedPrefix(
    pipeline: InferencePipeline,
    prompt: string,
    options: GenerateOptions = {}
  ): Promise<KVCacheSnapshot> {
    this.sharedPrefix = await pipeline.prefillKVOnly(prompt, options);
    return this.sharedPrefix;
  }

  getSharedPrefix(): KVCacheSnapshot | null {
    return this.sharedPrefix;
  }

  setSharedPrefix(snapshot: KVCacheSnapshot | null): void {
    this.sharedPrefix = snapshot;
  }

  clear(): void {
    this.sharedPrefix = null;
  }
}
