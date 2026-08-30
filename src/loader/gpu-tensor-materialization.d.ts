import type { Tensor } from '../gpu/tensor.js';

export declare function loadGpuTensor(
  loader: {
    _loadTensor(name: string, toGPU: boolean, silent: boolean): Promise<unknown>;
    tensorLocations: Map<string, { shape?: number[] }>;
  },
  name: string,
  silent?: boolean
): Promise<Tensor | null>;
