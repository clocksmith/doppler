import type { Tensor } from '../../tensor.js';
import type { QwenLinearAttentionPrepareOptions } from '../qwen_linear_attention_prepare.js';

export interface QwenLinearAttentionPrepareBackwardInputs {
  mixed: Tensor;
  a: Tensor;
  b: Tensor;
  aLog: Tensor;
  dtBias: Tensor;
  gradQuery: Tensor;
  gradKey: Tensor;
  gradValue: Tensor;
  gradLogDecay: Tensor;
  gradBeta: Tensor;
}

export interface QwenLinearAttentionPrepareGradients {
  mixed: Tensor;
  a: Tensor;
  b: Tensor;
}

export declare function runQwenLinearAttentionPrepareBackward(
  inputs: QwenLinearAttentionPrepareBackwardInputs,
  options: QwenLinearAttentionPrepareOptions
): Promise<QwenLinearAttentionPrepareGradients>;
