import type { Tensor } from '../../gpu/tensor.js';
import type {
  QwenDecoderMlpCache,
  QwenDecoderMlpInputs,
} from './qwen-decoder-mlp-training-module.js';
import type {
  QwenLinearAttentionTrainingModuleCache,
  QwenLinearAttentionTrainingModuleInputs,
  QwenLinearAttentionTrainingModuleOptions,
} from './qwen-linear-attention-training-core.js';

export interface QwenLinearDecoderLayerOptions
  extends QwenLinearAttentionTrainingModuleOptions {
  intermediateSize: number;
}

export interface QwenLinearDecoderLayerInputs {
  hidden: Tensor;
  inputNormWeight: Tensor;
  postAttentionNormWeight: Tensor;
  attention: Omit<QwenLinearAttentionTrainingModuleInputs, 'hidden'>;
  mlp: Omit<QwenDecoderMlpInputs, 'hidden'>;
}

export interface QwenLinearDecoderLayerCache {
  dims: Record<string, number>;
  inputNorm: Tensor;
  attentionInputs: QwenLinearAttentionTrainingModuleInputs;
  attentionOutput: Tensor;
  attentionCache: QwenLinearAttentionTrainingModuleCache;
  postAttention: Tensor;
  mlpCache: QwenDecoderMlpCache;
}

export declare function runQwenLinearDecoderLayerForward(
  inputs: QwenLinearDecoderLayerInputs,
  options: QwenLinearDecoderLayerOptions
): Promise<{ output: Tensor; finalState: Tensor; cache: QwenLinearDecoderLayerCache }>;

export declare function runQwenLinearDecoderLayerBackward(
  inputs: QwenLinearDecoderLayerInputs,
  gradOutput: Tensor,
  cache: QwenLinearDecoderLayerCache,
  options: QwenLinearDecoderLayerOptions
): Promise<{
  hidden: Tensor;
  initialState: Tensor;
  lora: {
    gate: { A: Tensor | null; B: Tensor | null };
    up: { A: Tensor | null; B: Tensor | null };
    down: { A: Tensor | null; B: Tensor | null };
  };
}>;

export declare function releaseQwenLinearDecoderLayerCache(
  cache: QwenLinearDecoderLayerCache
): void;
