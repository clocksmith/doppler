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

/** True when the selected-row f16 kernel can read this LM head directly. */
export declare function supportsLmHeadSelectLogitsF16(lmHead: unknown): boolean;

export declare function runLmHeadSelectLogitsF16(
  inputTensor: Tensor,
  lmHead: WeightBuffer,
  options: LmHeadSelectLogitsF16Options
): Promise<LmHeadSelectLogitsF16Result>;
