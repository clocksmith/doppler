import {
  qwenFrozenLoraProjectionBackward,
  qwenFrozenLoraProjectionForward,
  qwenFullAttentionModuleBackward,
  qwenFullAttentionModuleForward,
  qwenRmsNormOffsetBackward,
  qwenRmsNormOffsetForward,
} from './qwen-full-attention-reference.js';

function positiveInteger(value, label) {
  const parsed = Math.floor(Number(value));
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return parsed;
}

function add(left, right, label) {
  if (left.length !== right.length) {
    throw new Error(`${label} requires equal-length inputs.`);
  }
  return Float32Array.from(left, (value, index) => value + right[index]);
}

function siluGatedForward(gate, up) {
  if (gate.length !== up.length) throw new Error('gated SiLU requires equal-length inputs.');
  return Float32Array.from(gate, (value, index) => {
    const clamped = Math.max(-15, Math.min(15, value));
    return (value / (1 + Math.exp(-clamped))) * up[index];
  });
}

function siluGatedBackward(gate, up, gradOutput) {
  if (gate.length !== up.length || gate.length !== gradOutput.length) {
    throw new Error('gated SiLU backward requires equal-length inputs.');
  }
  const gradGate = new Float32Array(gate.length);
  const gradUp = new Float32Array(up.length);
  for (let index = 0; index < gate.length; index += 1) {
    const clamped = Math.max(-15, Math.min(15, gate[index]));
    const sigmoid = 1 / (1 + Math.exp(-clamped));
    gradGate[index] = gradOutput[index]
      * sigmoid * (1 + (gate[index] * (1 - sigmoid))) * up[index];
    gradUp[index] = gradOutput[index] * gate[index] * sigmoid;
  }
  return { gate: gradGate, up: gradUp };
}

function resolveDimensions(options) {
  const numTokens = positiveInteger(options?.numTokens, 'numTokens');
  const hiddenSize = positiveInteger(options?.hiddenSize, 'hiddenSize');
  const intermediateSize = positiveInteger(options?.intermediateSize, 'intermediateSize');
  return { numTokens, hiddenSize, intermediateSize };
}

function buildAttentionInputs(inputs, hidden) {
  return { ...inputs.attention, hidden };
}

export function qwenFullDecoderLayerForward(inputs, options) {
  const dims = resolveDimensions(options);
  const inputNorm = qwenRmsNormOffsetForward(
    inputs.hidden,
    inputs.inputNormWeight,
    dims.numTokens,
    dims.hiddenSize,
    options.rmsEps
  );
  const attentionInputs = buildAttentionInputs(inputs, inputNorm.output);
  const attention = qwenFullAttentionModuleForward(attentionInputs, options);
  const normalizedAttention = qwenRmsNormOffsetForward(
    attention.output,
    inputs.postAttentionNormWeight,
    dims.numTokens,
    dims.hiddenSize,
    options.rmsEps
  );
  const postAttention = add(inputs.hidden, normalizedAttention.output, 'attention residual');
  const gate = qwenFrozenLoraProjectionForward(
    postAttention,
    inputs.mlp.gateWeight,
    dims.numTokens,
    dims.hiddenSize,
    dims.intermediateSize,
    inputs.mlp.lora?.gate
  );
  const up = qwenFrozenLoraProjectionForward(
    postAttention,
    inputs.mlp.upWeight,
    dims.numTokens,
    dims.hiddenSize,
    dims.intermediateSize,
    inputs.mlp.lora?.up
  );
  const activated = siluGatedForward(gate.output, up.output);
  const down = qwenFrozenLoraProjectionForward(
    activated,
    inputs.mlp.downWeight,
    dims.numTokens,
    dims.intermediateSize,
    dims.hiddenSize,
    inputs.mlp.lora?.down
  );
  return {
    output: add(postAttention, down.output, 'MLP residual'),
    cache: {
      inputNorm,
      attentionInputs,
      attention,
      normalizedAttention,
      postAttention,
      gate,
      up,
      activated,
      down,
    },
  };
}

export function qwenFullDecoderLayerBackward(inputs, gradOutput, cache, options) {
  const dims = resolveDimensions(options);
  const downGradients = qwenFrozenLoraProjectionBackward(
    cache.activated,
    inputs.mlp.downWeight,
    gradOutput,
    dims.numTokens,
    dims.intermediateSize,
    dims.hiddenSize,
    inputs.mlp.lora?.down,
    cache.down.cache
  );
  const activationGradients = siluGatedBackward(
    cache.gate.output,
    cache.up.output,
    downGradients.input
  );
  const gateGradients = qwenFrozenLoraProjectionBackward(
    cache.postAttention,
    inputs.mlp.gateWeight,
    activationGradients.gate,
    dims.numTokens,
    dims.hiddenSize,
    dims.intermediateSize,
    inputs.mlp.lora?.gate,
    cache.gate.cache
  );
  const upGradients = qwenFrozenLoraProjectionBackward(
    cache.postAttention,
    inputs.mlp.upWeight,
    activationGradients.up,
    dims.numTokens,
    dims.hiddenSize,
    dims.intermediateSize,
    inputs.mlp.lora?.up,
    cache.up.cache
  );
  const mlpInputGradient = add(gateGradients.input, upGradients.input, 'MLP input gradient');
  const postAttentionGradient = add(gradOutput, mlpInputGradient, 'MLP residual gradient');
  const attentionOutputGradient = qwenRmsNormOffsetBackward(
    cache.attention.output,
    inputs.postAttentionNormWeight,
    postAttentionGradient,
    cache.normalizedAttention.cache,
    dims.numTokens,
    dims.hiddenSize
  );
  const attentionGradients = qwenFullAttentionModuleBackward(
    cache.attentionInputs,
    attentionOutputGradient,
    cache.attention.cache,
    options
  );
  const inputNormGradient = qwenRmsNormOffsetBackward(
    inputs.hidden,
    inputs.inputNormWeight,
    attentionGradients.hidden,
    cache.inputNorm.cache,
    dims.numTokens,
    dims.hiddenSize
  );
  return {
    hidden: add(postAttentionGradient, inputNormGradient, 'attention residual gradient'),
    lora: {
      q: attentionGradients.lora.q,
      k: attentionGradients.lora.k,
      v: attentionGradients.lora.v,
      o: attentionGradients.lora.o,
      gate: { A: gateGradients.A, B: gateGradients.B },
      up: { A: upGradients.A, B: upGradients.B },
      down: { A: downGradients.A, B: downGradients.B },
    },
  };
}
