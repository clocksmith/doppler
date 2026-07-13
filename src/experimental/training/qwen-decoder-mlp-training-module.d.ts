import type { Tensor } from '../../gpu/tensor.js';
import type { QwenFullAttentionLoraAdapter } from './qwen-full-attention-training-module.js';

export interface QwenDecoderMlpOptions {
  numTokens: number;
  hiddenSize: number;
  intermediateSize: number;
}

export interface QwenDecoderMlpInputs {
  hidden: Tensor;
  gateWeight: Tensor;
  upWeight: Tensor;
  downWeight: Tensor;
  lora?: {
    gate?: QwenFullAttentionLoraAdapter;
    up?: QwenFullAttentionLoraAdapter;
    down?: QwenFullAttentionLoraAdapter;
  };
}

export interface QwenDecoderMlpCache {
  dims: Record<string, number>;
  gate: Tensor;
  up: Tensor;
  activated: Tensor;
  projectionDowns: {
    gate: Tensor | null;
    up: Tensor | null;
    down: Tensor | null;
  };
}

export interface QwenDecoderMlpGradients {
  hidden: Tensor;
  lora: {
    gate: { A: Tensor | null; B: Tensor | null };
    up: { A: Tensor | null; B: Tensor | null };
    down: { A: Tensor | null; B: Tensor | null };
  };
}

export declare function runQwenDecoderMlpForward(
  inputs: QwenDecoderMlpInputs,
  options: QwenDecoderMlpOptions
): Promise<{ output: Tensor; cache: QwenDecoderMlpCache }>;

export declare function runQwenDecoderMlpBackward(
  inputs: QwenDecoderMlpInputs,
  gradOutput: Tensor,
  cache: QwenDecoderMlpCache,
  options: QwenDecoderMlpOptions
): Promise<QwenDecoderMlpGradients>;

export declare function releaseQwenDecoderMlpCache(cache: QwenDecoderMlpCache): void;
