import type { TensorLocation } from './loader-types.js';

export declare function getTensorNamesByRole(
  tensorLocations: Map<string, TensorLocation>,
  role: TensorLocation['role'],
  group?: string | null
): string[];
