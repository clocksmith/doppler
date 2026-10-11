import type { TensorLocation } from '../loader-types.js';
import type { TensorLoadConfig, TensorLoadResult } from './tensor-loader.js';
export declare function loadW8A8Packed(
  shardData: Uint8Array, location: TensorLocation, name: string, config: TensorLoadConfig
): Promise<TensorLoadResult>;
