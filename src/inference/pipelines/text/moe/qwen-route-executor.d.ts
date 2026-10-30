import type { Tensor } from '../../../../gpu/tensor.js';
import type { WeightBuffer } from '../../../../gpu/weight-buffer.js';

export interface QwenRouteExpertOptions {
  inputTensor: Tensor;
  indicesBuffer: GPUBuffer;
  weightsBuffer: GPUBuffer;
  weights: { numExperts: number; expertFormat: string; gateUp: WeightBuffer; down: WeightBuffer };
  expectedExpertFormat: string;
  numTokens: number;
  topK: number;
  hiddenSize: number;
  intermediateSize: number;
  activationDtype: string;
  hiddenActivation: string;
  swigluLimit: number | null;
}
export declare function runQwenRouteExperts(options: QwenRouteExpertOptions): Promise<Tensor>;
