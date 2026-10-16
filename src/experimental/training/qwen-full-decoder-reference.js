import { qwenFullAttentionModuleForward, qwenFullAttentionModuleBackward } from './qwen-full-attention-reference.js';
import { decoderLayerForward, decoderLayerBackward } from './qwen-decoder-reference.js';

export function qwenFullDecoderLayerForward(inputs, options) {
  return decoderLayerForward(inputs, options, qwenFullAttentionModuleForward);
}

export function qwenFullDecoderLayerBackward(inputs, gradOutput, cache, options) {
  const result = decoderLayerBackward(inputs, gradOutput, cache, options, qwenFullAttentionModuleBackward);
  const { q, k, v, o } = result.attentionGradients.lora;
  return { hidden: result.hidden, lora: { q, k, v, o, ...result.lora } };
}
