import {
  qwenFrozenLoraProjectionBackward,
  qwenFrozenLoraProjectionForward,
  qwenRmsNormOffsetBackward,
  qwenRmsNormOffsetForward,
} from './qwen-full-attention-reference.js';
import {
  qwenLinearAttentionModuleBackward,
  qwenLinearAttentionModuleForward,
} from './qwen-linear-attention-reference.js';

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
  return {
    numTokens: positiveInteger(options?.numTokens, 'numTokens'),
    hiddenSize: positiveInteger(options?.hiddenSize, 'hiddenSize'),
    intermediateSize: positiveInteger(options?.intermediateSize, 'intermediateSize'),
  };
}

function buildAttentionInputs(inputs, hidden) {
  return { ...inputs.attention, hidden };
}

export function qwenLinearDecoderLayerForward(inputs, options) {
  const dims = resolveDimensions(options);
  const inputNorm = qwenRmsNormOffsetForward(
    inputs.hidden,
    inputs.inputNormWeight,
    dims.numTokens,
    dims.hiddenSize,
    options.rmsEps
  );
  const attentionInputs = buildAttentionInputs(inputs, inputNorm.output);
  const attentionOptions = { ...options, eps: options.l2Eps };
  const attention = qwenLinearAttentionModuleForward(attentionInputs, attentionOptions);
  const postAttention = add(inputs.hidden, attention.output, 'attention residual');
  const normalizedPostAttention = qwenRmsNormOffsetForward(
    postAttention,
    inputs.postAttentionNormWeight,
    dims.numTokens,
    dims.hiddenSize,
    options.rmsEps
  );
  const gate = qwenFrozenLoraProjectionForward(
    normalizedPostAttention.output,
    inputs.mlp.gateWeight,
    dims.numTokens,
    dims.hiddenSize,
    dims.intermediateSize,
    inputs.mlp.lora?.gate
  );
  const up = qwenFrozenLoraProjectionForward(
    normalizedPostAttention.output,
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
    finalState: attention.finalState,
    cache: {
      inputNorm,
      attentionInputs,
      attention,
      postAttention,
      normalizedPostAttention,
      gate,
      up,
      activated,
      down,
    },
  };
}

export function qwenLinearDecoderLayerBackward(inputs, gradOutput, cache, options) {
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
    cache.normalizedPostAttention.output,
    inputs.mlp.gateWeight,
    activationGradients.gate,
    dims.numTokens,
    dims.hiddenSize,
    dims.intermediateSize,
    inputs.mlp.lora?.gate,
    cache.gate.cache
  );
  const upGradients = qwenFrozenLoraProjectionBackward(
    cache.normalizedPostAttention.output,
    inputs.mlp.upWeight,
    activationGradients.up,
    dims.numTokens,
    dims.hiddenSize,
    dims.intermediateSize,
    inputs.mlp.lora?.up,
    cache.up.cache
  );
  const mlpInputGradient = add(gateGradients.input, upGradients.input, 'MLP input gradient');
  const postAttentionMlpGradient = qwenRmsNormOffsetBackward(
    cache.postAttention,
    inputs.postAttentionNormWeight,
    mlpInputGradient,
    cache.normalizedPostAttention.cache,
    dims.numTokens,
    dims.hiddenSize
  );
  const postAttentionGradient = add(
    gradOutput,
    postAttentionMlpGradient,
    'MLP residual gradient'
  );
  const attentionGradients = qwenLinearAttentionModuleBackward(
    cache.attentionInputs,
    postAttentionGradient,
    cache.attention.cache,
    { ...options, eps: options.l2Eps }
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
    initialState: attentionGradients.initialState,
    lora: {
      gate: { A: gateGradients.A, B: gateGradients.B },
      up: { A: upGradients.A, B: upGradients.B },
      down: { A: downGradients.A, B: downGradients.B },
    },
  };
}
