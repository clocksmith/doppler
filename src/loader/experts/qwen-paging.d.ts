import type { ExpertLoaderContext } from './expert-loader.js';
import type { Qwen35PagerConfig } from '../../config/schema/loading.schema.js';
import type { CommandRecorder } from '../../gpu/command-recorder.js';

export interface QwenPagingReceipt {
  maxResidentLayers: number;
  residentLayers: number[];
  evictedLayers: number[];
}

/** Settle prior GPU work, retire complete packed banks and invalidate both alias maps. */
export declare function prepareQwenExpertLayer(
  context: Pick<ExpertLoaderContext, 'manifest' | 'experts' | 'gpuBuffers'>,
  layer: number,
  consumerExperts: Map<string, unknown>,
  policy: Qwen35PagerConfig,
  recorder: CommandRecorder | null,
  signal?: AbortSignal
): Promise<QwenPagingReceipt | null>;
