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
