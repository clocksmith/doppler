import type { TensorLocation } from '../loader-types.js';
import type { TensorLoadConfig, TensorLoadResult } from './tensor-loader.js';
export function loadFP8Packed(bytes: Uint8Array, location: TensorLocation, name: string,
  config: TensorLoadConfig): Promise<TensorLoadResult>;
