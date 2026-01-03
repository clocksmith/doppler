/**
 * Multi-model recorder for shared prefix KV caching.
 *
 * @module gpu/multi-model-recorder
 */

import type { InferencePipeline, KVCacheSnapshot, GenerateOptions } from '../inference/pipelines/text.js';

export declare class MultiModelRecorder {
  computeSharedPrefix(
    pipeline: InferencePipeline,
    prompt: string,
    options?: GenerateOptions
  ): Promise<KVCacheSnapshot>;

  getSharedPrefix(): KVCacheSnapshot | null;

  setSharedPrefix(snapshot: KVCacheSnapshot | null): void;

  clear(): void;
}
