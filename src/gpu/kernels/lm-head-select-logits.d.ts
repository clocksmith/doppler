import type { Tensor } from '../tensor.js';
import type { WeightBuffer } from '../weight-buffer.js';

export interface LmHeadSelectLogitsF16Options {
  device: GPUDevice;
  hiddenSize: number;
  vocabSize: number;
  tokenIds: readonly number[] | ArrayLike<number>;
  hiddenOffset: number;
  logitSoftcap: number;
}

export interface LmHeadSelectLogitsF16Result {
  outputBuffer: GPUBuffer;
  tokenIdBuffer: GPUBuffer;
  tokenIds: number[];
}

export declare function runLmHeadSelectLogitsF16(
  inputTensor: Tensor,
  lmHead: WeightBuffer,
  options: LmHeadSelectLogitsF16Options
): Promise<LmHeadSelectLogitsF16Result>;
