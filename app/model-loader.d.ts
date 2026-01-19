/**
 * model-loader.d.ts - Pipeline loader for app UI
 *
 * @module app/model-loader
 */

import type { ModelInfo } from './model-selector.js';
import type { Pipeline } from '../src/inference/pipeline.js';
import type { MemoryCapabilities } from '../src/memory/capability.js';

export type LoadSourceType = 'disk' | 'network' | 'cache';

export interface LoadProgress {
  phase: 'source' | 'gpu';
  percent: number;
  message?: string;
  bytesLoaded?: number;
  totalBytes?: number;
  speed?: number;
}

export interface ModelLoaderCallbacks {
  onProgress?: (progress: LoadProgress) => void;
  onSourceType?: (sourceType: LoadSourceType) => void;
}

export interface ModelLoadOptions {
  preferredSource?: 'server' | 'browser';
}

export declare class ModelLoader {
  constructor(callbacks?: ModelLoaderCallbacks);

  get pipeline(): Pipeline | null;
  get currentModel(): ModelInfo | null;
  get memoryCapabilities(): MemoryCapabilities | null;

  load(model: ModelInfo, options?: ModelLoadOptions): Promise<Pipeline>;
  unload(): Promise<void>;
  clearAllMemory(): Promise<void>;
  getMemoryStats(): unknown | null;
  getKVCacheStats(): unknown | null;
  clearKVCache(): void;
}
