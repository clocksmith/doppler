import type { DopplerLoader } from '../doppler-loader.js';
import type { ExpertLoaderContext } from './expert-loader.js';
import type { ExpertCacheConfigSchema } from '../../config/schema/loading.schema.js';
import type { CommandRecorder } from '../../gpu/command-recorder.js';
import type { QwenPagingReceipt } from './qwen-paging.js';
import type { RoutedExpertWeights } from './routed-expert-cache.js';
export declare function prepareExpertLayer(loader: DopplerLoader, policy: ExpertCacheConfigSchema,
  layer: number, consumers: Map<string, unknown>, recorder: CommandRecorder | null, signal?: AbortSignal,
  activeSelection?: string): Promise<QwenPagingReceipt | null>;
export declare function prepareRoutedExperts(loader: DopplerLoader, layer: number,
  experts: number[], signal?: AbortSignal): Promise<RoutedExpertWeights>;
export declare function disposeRoutedExperts(loader: DopplerLoader): Promise<unknown[]>;
export declare function createExpertLoaderContext(loader: {
  manifest: ExpertLoaderContext['manifest']; tensorLocations: ExpertLoaderContext['tensorLocations'];
  shardCache: ExpertLoaderContext['shardCache']; expertCache: ExpertLoaderContext['expertCache'];
  experts: ExpertLoaderContext['experts']; gpuBuffers: ExpertLoaderContext['gpuBuffers']; keepF32Weights: boolean;
  _getLoadShard(): ExpertLoaderContext['loadShard']; _loadTensor: ExpertLoaderContext['loadTensor'];
}): ExpertLoaderContext;
