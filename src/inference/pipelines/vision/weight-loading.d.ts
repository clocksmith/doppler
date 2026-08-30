import type { Tensor } from '../../../gpu/tensor.js';

export interface VisionGpuTensorLoader {
  loadGpuTensor(name: string, silent?: boolean): Promise<Tensor | null>;
}

export declare function loadRequiredVisionGpuTensor(
  loader: VisionGpuTensorLoader,
  name: string
): Promise<Tensor>;

export declare function loadVisionScalar(
  loadRequiredTensor: (name: string, toGPU?: boolean) => Promise<unknown>,
  name: string
): Promise<number>;

export declare function loadVisionClipRange(
  loadRequiredTensor: (name: string, toGPU?: boolean) => Promise<unknown>,
  prefix: string
): Promise<{ inputMin: number; inputMax: number; outputMin: number; outputMax: number }>;

export declare function loadGlmOcrVisionWeights(
  loader: VisionGpuTensorLoader,
  options: { textHiddenSize: number; depth: number }
): Promise<Record<string, unknown>>;
