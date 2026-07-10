import type { Tensor } from '../tensor.js';
import type { WeightBuffer } from '../weight-buffer.js';
import type { CommandRecorder } from '../command-recorder.js';

export interface LmHeadArgmaxOptions {
  vocabSize: number;
  hiddenSize: number;
  padTokenId: number | null;
  logitSoftcap: number;
  outputBuffer?: GPUBuffer | null;
  outputIndex: number;
}

export type LmHeadArgmaxF16Options = LmHeadArgmaxOptions;

export declare function recordLmHeadArgmax(
  recorder: CommandRecorder,
  inputTensor: Tensor,
  lmHead: WeightBuffer,
  options: LmHeadArgmaxOptions
): Promise<GPUBuffer>;

export declare function recordLmHeadArgmaxF16(
  recorder: CommandRecorder,
  inputTensor: Tensor,
  lmHead: WeightBuffer,
  options: LmHeadArgmaxF16Options
): Promise<GPUBuffer>;
