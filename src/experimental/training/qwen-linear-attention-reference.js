function requirePositiveInteger(value, label) {
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return value;
}

function requireArrayLength(value, expected, label) {
  if (!(value instanceof Float32Array) || value.length !== expected) {
    throw new Error(`${label} must be Float32Array(${expected}).`);
  }
  return value;
}

function silu(value) {
  const clamped = Math.max(-15, Math.min(15, value));
  const sigmoid = 1 / (1 + Math.exp(-clamped));
  return { value: value * sigmoid, derivative: sigmoid * (1 + (value * (1 - sigmoid))) };
}

export function causalConvSiluForward(input, weight, options) {
  const numTokens = requirePositiveInteger(options?.numTokens, 'numTokens');
  const channels = requirePositiveInteger(options?.channels, 'channels');
  const kernelSize = requirePositiveInteger(options?.kernelSize, 'kernelSize');
  requireArrayLength(input, numTokens * channels, 'input');
  requireArrayLength(weight, channels * kernelSize, 'weight');
  const raw = new Float32Array(input.length);
  const output = new Float32Array(input.length);
  for (let token = 0; token < numTokens; token += 1) {
    for (let channel = 0; channel < channels; channel += 1) {
      let sum = 0;
      for (let kernel = 0; kernel < kernelSize; kernel += 1) {
        const sourceToken = token + kernel - kernelSize + 1;
        if (sourceToken < 0) continue;
        sum += input[(sourceToken * channels) + channel]
          * weight[(channel * kernelSize) + kernel];
      }
      const offset = (token * channels) + channel;
      raw[offset] = sum;
      output[offset] = silu(sum).value;
    }
  }
  return { output, cache: { raw, numTokens, channels, kernelSize } };
}

export function causalConvSiluBackward(input, weight, gradOutput, cache, options) {
  const numTokens = requirePositiveInteger(options?.numTokens, 'numTokens');
  const channels = requirePositiveInteger(options?.channels, 'channels');
  const kernelSize = requirePositiveInteger(options?.kernelSize, 'kernelSize');
  requireArrayLength(input, numTokens * channels, 'input');
  requireArrayLength(weight, channels * kernelSize, 'weight');
  requireArrayLength(gradOutput, input.length, 'gradOutput');
  const raw = requireArrayLength(cache?.raw, input.length, 'cache.raw');
  if (cache.numTokens !== numTokens || cache.channels !== channels || cache.kernelSize !== kernelSize) {
    throw new Error('causal convolution backward cache dimensions do not match options.');
  }
  const gradInput = new Float32Array(input.length);
  const gradWeight = new Float32Array(weight.length);
  for (let token = 0; token < numTokens; token += 1) {
    for (let channel = 0; channel < channels; channel += 1) {
      const offset = (token * channels) + channel;
      const gradRaw = gradOutput[offset] * silu(raw[offset]).derivative;
      for (let kernel = 0; kernel < kernelSize; kernel += 1) {
        const sourceToken = token + kernel - kernelSize + 1;
        if (sourceToken < 0) continue;
        const sourceOffset = (sourceToken * channels) + channel;
        const weightOffset = (channel * kernelSize) + kernel;
        gradInput[sourceOffset] += gradRaw * weight[weightOffset];
        gradWeight[weightOffset] += gradRaw * input[sourceOffset];
      }
    }
  }
  return { input: gradInput, weight: gradWeight };
}

export function gatedRmsNormForward(input, gate, weight, options) {
  const rows = requirePositiveInteger(options?.rows, 'rows');
  const width = requirePositiveInteger(options?.width, 'width');
  const eps = Number(options?.eps);
  if (!Number.isFinite(eps) || eps <= 0) {
    throw new Error('eps must be finite and positive.');
  }
  requireArrayLength(input, rows * width, 'input');
  requireArrayLength(gate, rows * width, 'gate');
  requireArrayLength(weight, width, 'weight');
  const inverseRms = new Float32Array(rows);
  const output = new Float32Array(input.length);
  for (let row = 0; row < rows; row += 1) {
    let meanSquare = 0;
    for (let column = 0; column < width; column += 1) {
      const value = input[(row * width) + column];
      meanSquare += value * value;
    }
    const inverse = 1 / Math.sqrt((meanSquare / width) + eps);
    inverseRms[row] = inverse;
    for (let column = 0; column < width; column += 1) {
      const offset = (row * width) + column;
      output[offset] = input[offset] * inverse * weight[column] * silu(gate[offset]).value;
    }
  }
  return { output, cache: { inverseRms, rows, width, eps } };
}

export function gatedRmsNormBackward(input, gate, weight, gradOutput, cache, options) {
  const rows = requirePositiveInteger(options?.rows, 'rows');
  const width = requirePositiveInteger(options?.width, 'width');
  const eps = Number(options?.eps);
  requireArrayLength(input, rows * width, 'input');
  requireArrayLength(gate, rows * width, 'gate');
  requireArrayLength(weight, width, 'weight');
  requireArrayLength(gradOutput, input.length, 'gradOutput');
  const inverseRms = requireArrayLength(cache?.inverseRms, rows, 'cache.inverseRms');
  if (cache.rows !== rows || cache.width !== width || cache.eps !== eps) {
    throw new Error('gated RMSNorm backward cache dimensions do not match options.');
  }
  const gradInput = new Float32Array(input.length);
  const gradGate = new Float32Array(gate.length);
  const gradWeight = new Float32Array(weight.length);
  for (let row = 0; row < rows; row += 1) {
    const inverse = inverseRms[row];
    let inputGradientDot = 0;
    for (let column = 0; column < width; column += 1) {
      const offset = (row * width) + column;
      const activation = silu(gate[offset]);
      const gradNormalized = gradOutput[offset] * weight[column] * activation.value;
      inputGradientDot += gradNormalized * input[offset];
      gradGate[offset] = gradOutput[offset]
        * input[offset]
        * inverse
        * weight[column]
        * activation.derivative;
      gradWeight[column] += gradOutput[offset]
        * input[offset]
        * inverse
        * activation.value;
    }
    const correction = (inputGradientDot * inverse * inverse) / width;
    for (let column = 0; column < width; column += 1) {
      const offset = (row * width) + column;
      const activation = silu(gate[offset]);
      const gradNormalized = gradOutput[offset] * weight[column] * activation.value;
      gradInput[offset] = inverse * (gradNormalized - (input[offset] * correction));
    }
  }
  return { input: gradInput, gate: gradGate, weight: gradWeight };
}

export function l2NormalizeForward(input, options) {
  const rows = requirePositiveInteger(options?.rows, 'rows');
  const width = requirePositiveInteger(options?.width, 'width');
  const eps = Number(options?.eps);
  if (!Number.isFinite(eps) || eps <= 0) {
    throw new Error('eps must be finite and positive.');
  }
  requireArrayLength(input, rows * width, 'input');
  const inverseNorm = new Float32Array(rows);
  const output = new Float32Array(input.length);
  for (let row = 0; row < rows; row += 1) {
    let sumSquares = 0;
    for (let column = 0; column < width; column += 1) {
      const value = input[(row * width) + column];
      sumSquares += value * value;
    }
    const inverse = 1 / Math.sqrt(sumSquares + eps);
    inverseNorm[row] = inverse;
    for (let column = 0; column < width; column += 1) {
      const offset = (row * width) + column;
      output[offset] = input[offset] * inverse;
    }
  }
  return { output, cache: { inverseNorm, rows, width, eps } };
}

export function l2NormalizeBackward(input, gradOutput, cache, options) {
  const rows = requirePositiveInteger(options?.rows, 'rows');
  const width = requirePositiveInteger(options?.width, 'width');
  const eps = Number(options?.eps);
  requireArrayLength(input, rows * width, 'input');
  requireArrayLength(gradOutput, input.length, 'gradOutput');
  const inverseNorm = requireArrayLength(cache?.inverseNorm, rows, 'cache.inverseNorm');
  if (cache.rows !== rows || cache.width !== width || cache.eps !== eps) {
    throw new Error('L2 normalization backward cache dimensions do not match options.');
  }
  const gradInput = new Float32Array(input.length);
  for (let row = 0; row < rows; row += 1) {
    const inverse = inverseNorm[row];
    let dot = 0;
    for (let column = 0; column < width; column += 1) {
      const offset = (row * width) + column;
      dot += gradOutput[offset] * input[offset];
    }
    const correction = dot * inverse * inverse;
    for (let column = 0; column < width; column += 1) {
      const offset = (row * width) + column;
      gradInput[offset] = inverse * (gradOutput[offset] - (input[offset] * correction));
    }
  }
  return gradInput;
}

export function gatedDeltaParametersForward(a, b, aLog, dtBias) {
  if (!(a instanceof Float32Array) || !(b instanceof Float32Array)
    || !(aLog instanceof Float32Array) || !(dtBias instanceof Float32Array)
    || a.length !== b.length || a.length !== aLog.length || a.length !== dtBias.length) {
    throw new Error('gated-delta parameter arrays must be equal-length Float32Array values.');
  }
  const logDecay = new Float32Array(a.length);
  const beta = new Float32Array(a.length);
  for (let index = 0; index < a.length; index += 1) {
    const betaSigmoid = 1 / (1 + Math.exp(-Math.max(-15, Math.min(15, b[index]))));
    const softplusInput = a[index] + dtBias[index];
    const softplus = Math.log1p(Math.exp(-Math.abs(softplusInput))) + Math.max(softplusInput, 0);
    beta[index] = betaSigmoid;
    logDecay[index] = -Math.exp(aLog[index]) * softplus;
  }
  return { logDecay, beta };
}

export function gatedDeltaParametersBackward(a, b, aLog, dtBias, gradLogDecay, gradBeta) {
  const forward = gatedDeltaParametersForward(a, b, aLog, dtBias);
  requireArrayLength(gradLogDecay, a.length, 'gradLogDecay');
  requireArrayLength(gradBeta, a.length, 'gradBeta');
  const gradA = new Float32Array(a.length);
  const gradB = new Float32Array(b.length);
  const gradALog = new Float32Array(aLog.length);
  const gradDtBias = new Float32Array(dtBias.length);
  for (let index = 0; index < a.length; index += 1) {
    const softplusInput = a[index] + dtBias[index];
    const sigmoid = 1 / (1 + Math.exp(-Math.max(-15, Math.min(15, softplusInput))));
    const common = gradLogDecay[index] * -Math.exp(aLog[index]) * sigmoid;
    gradA[index] = common;
    gradDtBias[index] = common;
    gradALog[index] = gradLogDecay[index] * forward.logDecay[index];
    gradB[index] = gradBeta[index] * forward.beta[index] * (1 - forward.beta[index]);
  }
  return { a: gradA, b: gradB, aLog: gradALog, dtBias: gradDtBias };
}
