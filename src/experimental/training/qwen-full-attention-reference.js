import {
  computeAttentionBackwardData,
  computeAttentionSoftmaxData,
} from './attention-backward.js';

function positiveInteger(value, label) {
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return value;
}

function requireLength(value, expected, label) {
  if (!(value instanceof Float32Array) || value.length !== expected) {
    throw new Error(`${label} must be Float32Array(${expected}).`);
  }
  return value;
}

function stableSigmoid(value) {
  if (value >= 0) {
    const exponent = Math.exp(-value);
    return 1 / (1 + exponent);
  }
  const exponent = Math.exp(value);
  return exponent / (1 + exponent);
}

function resolveSplitDimensions(options) {
  const numTokens = positiveInteger(options?.numTokens, 'numTokens');
  const numHeads = positiveInteger(options?.numHeads, 'numHeads');
  const headDim = positiveInteger(options?.headDim, 'headDim');
  return { numTokens, numHeads, headDim, elements: numTokens * numHeads * headDim };
}

export function qwenAttentionSplitQGateForward(input, options) {
  const dims = resolveSplitDimensions(options);
  requireLength(input, dims.elements * 2, 'input');
  const query = new Float32Array(dims.elements);
  const gate = new Float32Array(dims.elements);
  for (let token = 0; token < dims.numTokens; token += 1) {
    for (let head = 0; head < dims.numHeads; head += 1) {
      const inputBase = ((token * dims.numHeads) + head) * dims.headDim * 2;
      const outputBase = ((token * dims.numHeads) + head) * dims.headDim;
      query.set(input.subarray(inputBase, inputBase + dims.headDim), outputBase);
      gate.set(
        input.subarray(inputBase + dims.headDim, inputBase + (dims.headDim * 2)),
        outputBase
      );
    }
  }
  return { query, gate };
}

export function qwenAttentionSplitQGateBackward(gradQuery, gradGate, options) {
  const dims = resolveSplitDimensions(options);
  requireLength(gradQuery, dims.elements, 'gradQuery');
  requireLength(gradGate, dims.elements, 'gradGate');
  const input = new Float32Array(dims.elements * 2);
  for (let token = 0; token < dims.numTokens; token += 1) {
    for (let head = 0; head < dims.numHeads; head += 1) {
      const inputBase = ((token * dims.numHeads) + head) * dims.headDim * 2;
      const outputBase = ((token * dims.numHeads) + head) * dims.headDim;
      input.set(gradQuery.subarray(outputBase, outputBase + dims.headDim), inputBase);
      input.set(
        gradGate.subarray(outputBase, outputBase + dims.headDim),
        inputBase + dims.headDim
      );
    }
  }
  return input;
}

export function sigmoidGateForward(input, gate) {
  if (!(input instanceof Float32Array) || !(gate instanceof Float32Array)
    || input.length !== gate.length) {
    throw new Error('sigmoid gate input and gate must be equal-length Float32Array values.');
  }
  return Float32Array.from(input, (value, index) => value * stableSigmoid(gate[index]));
}

export function sigmoidGateBackward(input, gate, gradOutput) {
  if (!(gradOutput instanceof Float32Array) || gradOutput.length !== input.length) {
    throw new Error('sigmoid gate gradOutput must match input length.');
  }
  const gradInput = new Float32Array(input.length);
  const gradGate = new Float32Array(gate.length);
  for (let index = 0; index < input.length; index += 1) {
    const probability = stableSigmoid(gate[index]);
    gradInput[index] = gradOutput[index] * probability;
    gradGate[index] = gradOutput[index] * input[index] * probability * (1 - probability);
  }
  return { input: gradInput, gate: gradGate };
}

function resolveRopeDimensions(options) {
  const numTokens = positiveInteger(options?.numTokens, 'numTokens');
  const numHeads = positiveInteger(options?.numHeads, 'numHeads');
  const headDim = positiveInteger(options?.headDim, 'headDim');
  const rotaryDim = positiveInteger(options?.rotaryDim, 'rotaryDim');
  const pairSpanDim = positiveInteger(options?.pairSpanDim, 'pairSpanDim');
  const startPos = Math.floor(Number(options?.startPos ?? 0));
  if (headDim % 2 !== 0 || rotaryDim % 2 !== 0 || pairSpanDim % 2 !== 0
    || rotaryDim > headDim || pairSpanDim < rotaryDim || pairSpanDim > headDim
    || startPos < 0) {
    throw new Error('invalid partial RoPE geometry.');
  }
  return {
    numTokens,
    numHeads,
    headDim,
    rotaryDim,
    pairSpanDim,
    startPos,
    interleaved: options?.interleaved === true,
    halfRotary: rotaryDim / 2,
  };
}

function ropeTransform(input, freqsCos, freqsSin, options, inverse) {
  const dims = resolveRopeDimensions(options);
  requireLength(input, dims.numTokens * dims.numHeads * dims.headDim, 'input');
  const frequencyElements = (dims.startPos + dims.numTokens) * dims.halfRotary;
  if (freqsCos.length < frequencyElements || freqsSin.length < frequencyElements) {
    throw new Error('partial RoPE frequency tables are too short.');
  }
  const output = new Float32Array(input);
  for (let token = 0; token < dims.numTokens; token += 1) {
    for (let head = 0; head < dims.numHeads; head += 1) {
      const base = ((token * dims.numHeads) + head) * dims.headDim;
      for (let pair = 0; pair < dims.halfRotary; pair += 1) {
        const first = dims.interleaved ? pair * 2 : pair;
        const second = dims.interleaved ? (pair * 2) + 1 : pair + (dims.pairSpanDim / 2);
        const frequency = ((dims.startPos + token) * dims.halfRotary) + pair;
        const cos = freqsCos[frequency];
        const sin = freqsSin[frequency];
        const firstValue = input[base + first];
        const secondValue = input[base + second];
        if (inverse) {
          output[base + first] = (firstValue * cos) + (secondValue * sin);
          output[base + second] = (-firstValue * sin) + (secondValue * cos);
        } else {
          output[base + first] = (firstValue * cos) - (secondValue * sin);
          output[base + second] = (firstValue * sin) + (secondValue * cos);
        }
      }
    }
  }
  return output;
}

export function partialRopeForward(input, freqsCos, freqsSin, options) {
  return ropeTransform(input, freqsCos, freqsSin, options, false);
}

export function partialRopeBackward(gradOutput, freqsCos, freqsSin, options) {
  return ropeTransform(gradOutput, freqsCos, freqsSin, options, true);
}

function matmulRightTransposed(input, weight, rows, inputSize, outputSize) {
  requireLength(input, rows * inputSize, 'matmul input');
  requireLength(weight, outputSize * inputSize, 'matmul weight');
  const output = new Float32Array(rows * outputSize);
  for (let row = 0; row < rows; row += 1) {
    for (let outputIndex = 0; outputIndex < outputSize; outputIndex += 1) {
      let sum = 0;
      for (let inputIndex = 0; inputIndex < inputSize; inputIndex += 1) {
        sum += input[(row * inputSize) + inputIndex]
          * weight[(outputIndex * inputSize) + inputIndex];
      }
      output[(row * outputSize) + outputIndex] = sum;
    }
  }
  return output;
}

function matmulInputGradient(gradOutput, weight, rows, inputSize, outputSize) {
  requireLength(gradOutput, rows * outputSize, 'matmul gradOutput');
  const output = new Float32Array(rows * inputSize);
  for (let row = 0; row < rows; row += 1) {
    for (let inputIndex = 0; inputIndex < inputSize; inputIndex += 1) {
      let sum = 0;
      for (let outputIndex = 0; outputIndex < outputSize; outputIndex += 1) {
        sum += gradOutput[(row * outputSize) + outputIndex]
          * weight[(outputIndex * inputSize) + inputIndex];
      }
      output[(row * inputSize) + inputIndex] = sum;
    }
  }
  return output;
}

function rmsNormOffsetForward(input, weight, rows, width, eps) {
  requireLength(input, rows * width, 'RMSNorm input');
  requireLength(weight, width, 'RMSNorm weight');
  const output = new Float32Array(input.length);
  const inverseRms = new Float32Array(rows);
  for (let row = 0; row < rows; row += 1) {
    let sumSquares = 0;
    const base = row * width;
    for (let column = 0; column < width; column += 1) {
      sumSquares += input[base + column] ** 2;
    }
    const inverse = 1 / Math.sqrt((sumSquares / width) + eps);
    inverseRms[row] = inverse;
    for (let column = 0; column < width; column += 1) {
      output[base + column] = input[base + column] * inverse * (1 + weight[column]);
    }
  }
  return { output, cache: { inverseRms } };
}

function rmsNormOffsetBackward(input, weight, gradOutput, cache, rows, width) {
  const output = new Float32Array(input.length);
  for (let row = 0; row < rows; row += 1) {
    const base = row * width;
    const inverse = cache.inverseRms[row];
    let dot = 0;
    for (let column = 0; column < width; column += 1) {
      dot += gradOutput[base + column] * (1 + weight[column]) * input[base + column];
    }
    const correction = (dot * inverse * inverse) / width;
    for (let column = 0; column < width; column += 1) {
      const scaledGradient = gradOutput[base + column] * (1 + weight[column]);
      output[base + column] = inverse
        * (scaledGradient - (input[base + column] * correction));
    }
  }
  return output;
}

function attentionOutput(query, key, value, options) {
  const softmax = computeAttentionSoftmaxData(query, key, options);
  const output = new Float32Array(options.seqLen * options.numHeads * options.headDim);
  const headsPerKv = options.numHeads / options.numKVHeads;
  for (let token = 0; token < options.seqLen; token += 1) {
    for (let head = 0; head < options.numHeads; head += 1) {
      const kvHead = Math.floor(head / headsPerKv);
      const scoreBase = ((head * options.seqLen) + token) * options.seqLen;
      const outputBase = ((token * options.numHeads) + head) * options.headDim;
      for (let keyToken = 0; keyToken < options.seqLen; keyToken += 1) {
        const probability = softmax[scoreBase + keyToken];
        const valueBase = ((keyToken * options.numKVHeads) + kvHead) * options.headDim;
        for (let dim = 0; dim < options.headDim; dim += 1) {
          output[outputBase + dim] += probability * value[valueBase + dim];
        }
      }
    }
  }
  return { output, softmax };
}

function resolveModuleDimensions(options) {
  const numTokens = positiveInteger(options?.numTokens, 'numTokens');
  const hiddenSize = positiveInteger(options?.hiddenSize, 'hiddenSize');
  const numHeads = positiveInteger(options?.numHeads, 'numHeads');
  const numKVHeads = positiveInteger(options?.numKVHeads, 'numKVHeads');
  const headDim = positiveInteger(options?.headDim, 'headDim');
  const rmsEps = Number(options?.rmsEps);
  if (numHeads % numKVHeads !== 0 || !Number.isFinite(rmsEps) || rmsEps <= 0) {
    throw new Error('invalid Qwen full-attention module geometry.');
  }
  return {
    ...resolveRopeDimensions(options),
    numTokens,
    hiddenSize,
    numHeads,
    numKVHeads,
    headDim,
    rmsEps,
    querySize: numHeads * headDim,
    kvSize: numKVHeads * headDim,
  };
}

export function qwenFullAttentionModuleForward(inputs, options) {
  const dims = resolveModuleDimensions(options);
  const qProjection = matmulRightTransposed(
    inputs.hidden,
    inputs.qWeight,
    dims.numTokens,
    dims.hiddenSize,
    dims.querySize * 2
  );
  const split = qwenAttentionSplitQGateForward(qProjection, dims);
  const kProjection = matmulRightTransposed(
    inputs.hidden,
    inputs.kWeight,
    dims.numTokens,
    dims.hiddenSize,
    dims.kvSize
  );
  const value = matmulRightTransposed(
    inputs.hidden,
    inputs.vWeight,
    dims.numTokens,
    dims.hiddenSize,
    dims.kvSize
  );
  const queryNorm = rmsNormOffsetForward(
    split.query,
    inputs.qNormWeight,
    dims.numTokens * dims.numHeads,
    dims.headDim,
    dims.rmsEps
  );
  const keyNorm = rmsNormOffsetForward(
    kProjection,
    inputs.kNormWeight,
    dims.numTokens * dims.numKVHeads,
    dims.headDim,
    dims.rmsEps
  );
  const query = partialRopeForward(queryNorm.output, inputs.cos, inputs.sin, dims);
  const key = partialRopeForward(keyNorm.output, inputs.cos, inputs.sin, {
    ...dims,
    numHeads: dims.numKVHeads,
  });
  const attentionOptions = {
    seqLen: dims.numTokens,
    numHeads: dims.numHeads,
    numKVHeads: dims.numKVHeads,
    headDim: dims.headDim,
    scale: 1 / Math.sqrt(dims.headDim),
    causal: true,
  };
  const attention = attentionOutput(query, key, value, attentionOptions);
  const gated = sigmoidGateForward(attention.output, split.gate);
  const output = matmulRightTransposed(
    gated,
    inputs.oWeight,
    dims.numTokens,
    dims.querySize,
    dims.hiddenSize
  );
  return {
    output,
    cache: {
      dims,
      split,
      kProjection,
      value,
      queryNorm,
      keyNorm,
      query,
      key,
      attention,
      gated,
      attentionOptions,
    },
  };
}

export function qwenFullAttentionModuleBackward(inputs, gradOutput, cache, options) {
  const dims = resolveModuleDimensions(options);
  const gradGated = matmulInputGradient(
    gradOutput,
    inputs.oWeight,
    dims.numTokens,
    dims.querySize,
    dims.hiddenSize
  );
  const gateGradients = sigmoidGateBackward(
    cache.attention.output,
    cache.split.gate,
    gradGated
  );
  const attentionGradients = computeAttentionBackwardData(
    cache.query,
    cache.key,
    cache.value,
    cache.attention.softmax,
    gateGradients.input,
    cache.attentionOptions
  );
  const gradQueryNorm = partialRopeBackward(
    attentionGradients.dQ,
    inputs.cos,
    inputs.sin,
    dims
  );
  const gradKeyNorm = partialRopeBackward(
    attentionGradients.dK,
    inputs.cos,
    inputs.sin,
    { ...dims, numHeads: dims.numKVHeads }
  );
  const gradQuery = rmsNormOffsetBackward(
    cache.split.query,
    inputs.qNormWeight,
    gradQueryNorm,
    cache.queryNorm.cache,
    dims.numTokens * dims.numHeads,
    dims.headDim
  );
  const gradKey = rmsNormOffsetBackward(
    cache.kProjection,
    inputs.kNormWeight,
    gradKeyNorm,
    cache.keyNorm.cache,
    dims.numTokens * dims.numKVHeads,
    dims.headDim
  );
  const gradQProjection = qwenAttentionSplitQGateBackward(
    gradQuery,
    gateGradients.gate,
    dims
  );
  const contributions = [
    matmulInputGradient(
      gradQProjection,
      inputs.qWeight,
      dims.numTokens,
      dims.hiddenSize,
      dims.querySize * 2
    ),
    matmulInputGradient(
      gradKey,
      inputs.kWeight,
      dims.numTokens,
      dims.hiddenSize,
      dims.kvSize
    ),
    matmulInputGradient(
      attentionGradients.dV,
      inputs.vWeight,
      dims.numTokens,
      dims.hiddenSize,
      dims.kvSize
    ),
  ];
  const hidden = new Float32Array(dims.numTokens * dims.hiddenSize);
  for (const contribution of contributions) {
    for (let index = 0; index < hidden.length; index += 1) hidden[index] += contribution[index];
  }
  return { hidden };
}
