import { qwenLinearAttentionModuleForward, qwenLinearAttentionModuleBackward } from './qwen-linear-attention-reference.js';
import { decoderLayerForward, decoderLayerBackward } from './qwen-decoder-reference.js';

export function qwenLinearDecoderLayerForward(inputs, options) {
  const result = decoderLayerForward(inputs, options, (attentionInputs) =>
    qwenLinearAttentionModuleForward(attentionInputs, { ...options, eps: options.l2Eps }));
  return { output: result.output, finalState: result.cache.attention.finalState, cache: result.cache };
}

export function qwenLinearDecoderLayerBackward(inputs, gradOutput, cache, options) {
  const result = decoderLayerBackward(inputs, gradOutput, cache, options, (attentionInputs, gradient, attentionCache) =>
    qwenLinearAttentionModuleBackward(attentionInputs, gradient, attentionCache, { ...options, eps: options.l2Eps }));
  return { hidden: result.hidden, initialState: result.attentionGradients.initialState, lora: result.lora };
}
