import type { Tensor } from '../../gpu/tensor.js';
import type { QwenHybridLayer } from './qwen-hybrid-decoder-training-module.js';

export interface QwenCheckpointedHybridDecoderCache {
  checkpointInterval: number;
  layerCount: number;
  layerTypes: Array<'linear_attention' | 'full_attention'>;
  segmentTypes: Array<Array<'linear_attention' | 'full_attention'>>;
  layers: QwenHybridLayer[];
  checkpoints: Array<{
    startLayer: number;
    endLayer: number;
    hidden: Tensor;
    ownsHidden: boolean;
  }>;
}

export declare function runQwenCheckpointedHybridDecoderForward(
  inputs: { hidden: Tensor; layers: QwenHybridLayer[] },
  options: { checkpointInterval: number }
): Promise<{
  output: Tensor;
  finalStates: [];
  cache: QwenCheckpointedHybridDecoderCache;
}>;

export declare function runQwenCheckpointedHybridDecoderBackward(
  gradOutput: Tensor,
  cache: QwenCheckpointedHybridDecoderCache
): Promise<{
  hidden: Tensor;
  layers: Array<{
    type: 'linear_attention' | 'full_attention';
    lora: Record<string, { A: Tensor | null; B: Tensor | null }>;
    initialState: Tensor | null;
  }>;
}>;

export declare function releaseQwenCheckpointedHybridDecoderCache(
  cache: QwenCheckpointedHybridDecoderCache
): void;
