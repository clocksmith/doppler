import type { Tensor } from '../gpu/tensor.js';
import type { AutogradTape } from './autograd.js';

export interface LoraAdapterConfig {
  inDim: number;
  outDim: number;
  rank: number;
  alpha: number;
}

export declare class LoraAdapter {
  constructor(config: LoraAdapterConfig);
  A: Tensor;
  B: Tensor;
  alpha: number;
  rank: number;
  forward(input: Tensor, tape: AutogradTape): Promise<Tensor>;
  dispose(): void;
}
