import type { Tensor } from '../../tensor.js';
import type { QwenAttentionSplitQGateOptions } from '../qwen_attention_split_q_gate.js';

export declare function runQwenAttentionSplitQGateBackward(
  gradQuery: Tensor,
  gradGate: Tensor,
  options: QwenAttentionSplitQGateOptions
): Promise<Tensor>;
