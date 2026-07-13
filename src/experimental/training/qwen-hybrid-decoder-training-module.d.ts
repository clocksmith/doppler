import type { Tensor } from '../../gpu/tensor.js';
import type {
  QwenFullDecoderLayerCache,
  QwenFullDecoderLayerInputs,
  QwenFullDecoderLayerLoraGradients,
  QwenFullDecoderLayerOptions,
} from './qwen-full-decoder-training-module.js';
import type {
  QwenLinearDecoderLayerCache,
  QwenLinearDecoderLayerInputs,
  QwenLinearDecoderLayerOptions,
} from './qwen-linear-decoder-training-module.js';

export interface QwenHybridLinearLayer {
  type: 'linear_attention';
  inputs: Omit<QwenLinearDecoderLayerInputs, 'hidden'>;
  options: QwenLinearDecoderLayerOptions;
}

export interface QwenHybridFullLayer {
  type: 'full_attention';
  inputs: Omit<QwenFullDecoderLayerInputs, 'hidden'>;
  options: QwenFullDecoderLayerOptions;
}

export type QwenHybridLayer = QwenHybridLinearLayer | QwenHybridFullLayer;

export interface QwenHybridDecoderInputs {
  hidden: Tensor;
  layers: QwenHybridLayer[];
}

export interface QwenHybridDecoderCacheEntry {
  type: 'linear_attention' | 'full_attention';
  inputs: QwenLinearDecoderLayerInputs | QwenFullDecoderLayerInputs;
  options: QwenLinearDecoderLayerOptions | QwenFullDecoderLayerOptions;
  output: Tensor;
  cache: QwenLinearDecoderLayerCache | QwenFullDecoderLayerCache;
}

export interface QwenHybridDecoderCache {
  layerTypes: Array<'linear_attention' | 'full_attention'>;
  entries: QwenHybridDecoderCacheEntry[];
}

export declare function runQwenHybridDecoderForward(
  inputs: QwenHybridDecoderInputs
): Promise<{
  output: Tensor;
  finalStates: Array<{ layerIndex: number; state: Tensor }>;
  cache: QwenHybridDecoderCache;
}>;

export declare function runQwenHybridDecoderBackward(
  gradOutput: Tensor,
  cache: QwenHybridDecoderCache
): Promise<{
  hidden: Tensor;
  layers: Array<{
    type: 'linear_attention' | 'full_attention';
    lora: QwenFullDecoderLayerLoraGradients | {
      gate: { A: Tensor | null; B: Tensor | null };
      up: { A: Tensor | null; B: Tensor | null };
      down: { A: Tensor | null; B: Tensor | null };
    };
    initialState: Tensor | null;
  }>;
}>;

export declare function releaseQwenHybridDecoderCache(cache: QwenHybridDecoderCache): void;
