import type { Tensor } from '../../../../gpu/tensor.js';
import type { QwenRouteExpertOptions } from './qwen-route-executor.js';
import type { DopplerLoader } from '../../../../loader/doppler-loader.js';
import type { Qwen35RoutedConfig } from '../../../../config/schema/loading.schema.js';
export declare function executeRoutedExpertChunks(
  options: Omit<QwenRouteExpertOptions, 'weights'>,
  loader: Pick<DopplerLoader, 'prepareRoutedExperts'> & Partial<Pick<DopplerLoader, 'recordRoutedReadback'>>,
  policy: Qwen35RoutedConfig,
  signal?: AbortSignal
): Promise<Tensor>;
