import type { TensorLocation } from '../loader-types.js';
import type { TensorLoadConfig, TensorLoadResult } from './tensor-loader.js';

export declare function loadBF16(
  shardData: Uint8Array | GPUBuffer,
  location: TensorLocation,
  name: string,
  config: TensorLoadConfig
): Promise<TensorLoadResult>;
