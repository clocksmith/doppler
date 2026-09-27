import type { DopplerLoader } from './doppler-loader.js';
import type { LoadOptions, ShardLoadOptions } from './loader-types.js';
import type { LoadingConfigSchema } from '../config/schema/loading.schema.js';
export interface ModelLoadOwner extends DopplerLoader {
  _memoryMonitor: object | null;
  _loadingConfig: LoadingConfigSchema;
  _startMemoryLogging(): void;
  _stopMemoryLogging(status: string): void;
  _assertResidentBudget(stage: string): void;
  _buildTensorLocations(): Promise<void>;
  _loadShard(index: number, options?: ShardLoadOptions): Promise<ArrayBuffer>;
  _loadShardOverride: ModelLoadOwner['_loadShard'] | null;
  _loadEmbeddings(progress: LoadOptions['onProgress'] | null): Promise<void>;
  _loadLayer(index: number, progress: LoadOptions['onProgress'] | null): Promise<void>;
  _loadFinalWeights(progress: LoadOptions['onProgress'] | null): Promise<void>;
  _prefetchLayerShards(index: number): void;
}
export declare function load(this: ModelLoadOwner, modelId: string, options: LoadOptions): Promise<Record<string, unknown>>;
