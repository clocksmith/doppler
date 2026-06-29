import type { TensorLocation } from '../loader-types.js';
import type { TensorLoadConfig, TensorLoadResult } from './tensor-loader.js';

export declare function loadFunctionalDescriptor(
  shardData: Uint8Array & { descriptorShards?: Map<string, Uint8Array> },
  location: TensorLocation,
  name: string,
  config?: TensorLoadConfig
): Promise<TensorLoadResult>;
