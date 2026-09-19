import type { Tensor } from '../../gpu/tensor.js';
import type {
  QwenFullAttentionLoraAdapter,
  QwenFullAttentionLoraGradients,
  QwenFullAttentionTrainingModuleCache,
  QwenFullAttentionTrainingModuleInputs,
  QwenFullAttentionTrainingModuleOptions,
} from './qwen-full-attention-training-module.js';
import type { QwenDecoderMlpCache } from './qwen-decoder-mlp-training-module.js';

export interface QwenFullDecoderLayerOptions extends QwenFullAttentionTrainingModuleOptions {
  intermediateSize: number;
}

export interface QwenFullDecoderLayerInputs {
  hidden: Tensor;
  inputNormWeight: Tensor;
  postAttentionNormWeight: Tensor;
  attention: Omit<QwenFullAttentionTrainingModuleInputs, 'hidden'>;
  mlp: {
    gateWeight: Tensor;
    upWeight: Tensor;
    downWeight: Tensor;
    lora?: {
      gate?: QwenFullAttentionLoraAdapter;
      up?: QwenFullAttentionLoraAdapter;
      down?: QwenFullAttentionLoraAdapter;
    };
  };
}

export interface QwenFullDecoderLayerCache {
  dims: Record<string, number>;
  inputNorm: Tensor;
  attentionInputs: QwenFullAttentionTrainingModuleInputs;
  attentionOutput: Tensor;
  attentionCache: QwenFullAttentionTrainingModuleCache;
  postAttention: Tensor;
  normalizedPostAttention: Tensor;
  mlpCache: QwenDecoderMlpCache;
}

export interface QwenFullDecoderLayerLoraGradients extends QwenFullAttentionLoraGradients {
  gate: { A: Tensor | null; B: Tensor | null };
  up: { A: Tensor | null; B: Tensor | null };
  down: { A: Tensor | null; B: Tensor | null };
}

export declare function runQwenFullDecoderLayerForward(
  inputs: QwenFullDecoderLayerInputs,
  options: QwenFullDecoderLayerOptions
): Promise<{ output: Tensor; cache: QwenFullDecoderLayerCache }>;

export declare function runQwenFullDecoderLayerBackward(
  inputs: QwenFullDecoderLayerInputs,
  gradOutput: Tensor,
  cache: QwenFullDecoderLayerCache,
  options: QwenFullDecoderLayerOptions
): Promise<{ hidden: Tensor; lora: QwenFullDecoderLayerLoraGradients }>;

export declare function releaseQwenFullDecoderLayerCache(
  cache: QwenFullDecoderLayerCache
): void;
