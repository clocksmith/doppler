import {
  gatedDeltaRecurrentCheckpointedBackward,
  gatedDeltaRecurrentCheckpointedForward,
} from './qwen-gated-delta-reference.js';

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
  const sigmoid = value >= 0
    ? 1 / (1 + Math.exp(-value))
    : (() => {
        const exponent = Math.exp(value);
        return exponent / (1 + exponent);
      })();
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

function resolvePrepareDimensions(options) {
  const numTokens = requirePositiveInteger(options?.numTokens, 'numTokens');
  const numKeyHeads = requirePositiveInteger(options?.numKeyHeads, 'numKeyHeads');
  const numValueHeads = requirePositiveInteger(options?.numValueHeads, 'numValueHeads');
  const keyDim = requirePositiveInteger(options?.keyDim, 'keyDim');
  const valueDim = requirePositiveInteger(options?.valueDim, 'valueDim');
  const eps = Number(options?.eps);
  if (numValueHeads % numKeyHeads !== 0) {
    throw new Error('numValueHeads must be divisible by numKeyHeads.');
  }
  if (!Number.isFinite(eps) || eps <= 0) {
    throw new Error('eps must be finite and positive.');
  }
  const repeatFactor = numValueHeads / numKeyHeads;
  const querySize = numKeyHeads * keyDim;
  const keySize = querySize;
  const valueSize = numValueHeads * valueDim;
  return {
    numTokens,
    numKeyHeads,
    numValueHeads,
    keyDim,
    valueDim,
    eps,
    repeatFactor,
    querySize,
    keySize,
    valueSize,
    convSize: querySize + keySize + valueSize,
  };
}

export function qwenLinearAttentionPrepareForward(inputs, options) {
  const dims = resolvePrepareDimensions(options);
  const {
    numTokens,
    numKeyHeads,
    numValueHeads,
    keyDim,
    valueDim,
    eps,
    repeatFactor,
    querySize,
    keySize,
    valueSize,
    convSize,
  } = dims;
  const mixed = requireArrayLength(inputs?.mixed, numTokens * convSize, 'mixed');
  const a = requireArrayLength(inputs?.a, numTokens * numValueHeads, 'a');
  const b = requireArrayLength(inputs?.b, numTokens * numValueHeads, 'b');
  const aLog = requireArrayLength(inputs?.aLog, numValueHeads, 'aLog');
  const dtBias = requireArrayLength(inputs?.dtBias, numValueHeads, 'dtBias');
  const query = new Float32Array(numTokens * numValueHeads * keyDim);
  const key = new Float32Array(query.length);
  const value = new Float32Array(numTokens * valueSize);
  const logDecay = new Float32Array(numTokens * numValueHeads);
  const beta = new Float32Array(logDecay.length);

  for (let token = 0; token < numTokens; token += 1) {
    const mixedBase = token * convSize;
    for (let keyHead = 0; keyHead < numKeyHeads; keyHead += 1) {
      const rawQueryBase = mixedBase + (keyHead * keyDim);
      const rawKeyBase = mixedBase + querySize + (keyHead * keyDim);
      let querySumSquares = 0;
      let keySumSquares = 0;
      for (let dim = 0; dim < keyDim; dim += 1) {
        querySumSquares += mixed[rawQueryBase + dim] ** 2;
        keySumSquares += mixed[rawKeyBase + dim] ** 2;
      }
      const queryInverse = 1 / Math.sqrt(querySumSquares + eps);
      const keyInverse = 1 / Math.sqrt(keySumSquares + eps);
      for (let repeat = 0; repeat < repeatFactor; repeat += 1) {
        const valueHead = (keyHead * repeatFactor) + repeat;
        const outputBase = ((token * numValueHeads) + valueHead) * keyDim;
        for (let dim = 0; dim < keyDim; dim += 1) {
          query[outputBase + dim] = mixed[rawQueryBase + dim] * queryInverse;
          key[outputBase + dim] = mixed[rawKeyBase + dim] * keyInverse;
        }
      }
    }
    value.set(
      mixed.subarray(mixedBase + querySize + keySize, mixedBase + convSize),
      token * valueSize
    );
    for (let head = 0; head < numValueHeads; head += 1) {
      const scalarIndex = (token * numValueHeads) + head;
      const betaValue = 1 / (1 + Math.exp(-b[scalarIndex]));
      const softplusInput = a[scalarIndex] + dtBias[head];
      const softplus = Math.log1p(Math.exp(-Math.abs(softplusInput)))
        + Math.max(softplusInput, 0);
      beta[scalarIndex] = betaValue;
      logDecay[scalarIndex] = -Math.exp(aLog[head]) * softplus;
    }
  }
  return { query, key, value, logDecay, beta, cache: { dims } };
}

export function qwenLinearAttentionPrepareBackward(inputs, gradients, cache, options) {
  const dims = resolvePrepareDimensions(options);
  for (const key of Object.keys(dims)) {
    if (cache?.dims?.[key] !== dims[key]) {
      throw new Error(`linear-attention prepare cache mismatch for ${key}.`);
    }
  }
  const {
    numTokens,
    numKeyHeads,
    numValueHeads,
    keyDim,
    valueDim,
    eps,
    repeatFactor,
    querySize,
    keySize,
    valueSize,
    convSize,
  } = dims;
  const mixed = requireArrayLength(inputs?.mixed, numTokens * convSize, 'mixed');
  const a = requireArrayLength(inputs?.a, numTokens * numValueHeads, 'a');
  const b = requireArrayLength(inputs?.b, numTokens * numValueHeads, 'b');
  const aLog = requireArrayLength(inputs?.aLog, numValueHeads, 'aLog');
  const dtBias = requireArrayLength(inputs?.dtBias, numValueHeads, 'dtBias');
  const gradQuery = requireArrayLength(
    gradients?.query,
    numTokens * numValueHeads * keyDim,
    'gradients.query'
  );
  const gradKey = requireArrayLength(gradients?.key, gradQuery.length, 'gradients.key');
  const gradValue = requireArrayLength(
    gradients?.value,
    numTokens * numValueHeads * valueDim,
    'gradients.value'
  );
  const gradLogDecay = requireArrayLength(
    gradients?.logDecay,
    numTokens * numValueHeads,
    'gradients.logDecay'
  );
  const gradBeta = requireArrayLength(gradients?.beta, gradLogDecay.length, 'gradients.beta');
  const gradMixed = new Float32Array(mixed.length);
  const gradA = new Float32Array(a.length);
  const gradB = new Float32Array(b.length);
  const gradALog = new Float32Array(aLog.length);
  const gradDtBias = new Float32Array(dtBias.length);

  for (let token = 0; token < numTokens; token += 1) {
    const mixedBase = token * convSize;
    for (let keyHead = 0; keyHead < numKeyHeads; keyHead += 1) {
      for (const [rawBase, repeatedGradient] of [
        [mixedBase + (keyHead * keyDim), gradQuery],
        [mixedBase + querySize + (keyHead * keyDim), gradKey],
      ]) {
        const aggregated = new Float32Array(keyDim);
        let sumSquares = 0;
        let dot = 0;
        for (let dim = 0; dim < keyDim; dim += 1) {
          const rawValue = mixed[rawBase + dim];
          sumSquares += rawValue * rawValue;
          for (let repeat = 0; repeat < repeatFactor; repeat += 1) {
            const valueHead = (keyHead * repeatFactor) + repeat;
            const gradientBase = ((token * numValueHeads) + valueHead) * keyDim;
            aggregated[dim] += repeatedGradient[gradientBase + dim];
          }
          dot += aggregated[dim] * rawValue;
        }
        const inverse = 1 / Math.sqrt(sumSquares + eps);
        const correction = dot * inverse * inverse;
        for (let dim = 0; dim < keyDim; dim += 1) {
          gradMixed[rawBase + dim] = inverse
            * (aggregated[dim] - (mixed[rawBase + dim] * correction));
        }
      }
    }
    gradMixed.set(gradValue.subarray(token * valueSize, (token + 1) * valueSize),
      mixedBase + querySize + keySize);
    for (let head = 0; head < numValueHeads; head += 1) {
      const scalarIndex = (token * numValueHeads) + head;
      const betaValue = 1 / (1 + Math.exp(-b[scalarIndex]));
      const softplusInput = a[scalarIndex] + dtBias[head];
      const softplus = Math.log1p(Math.exp(-Math.abs(softplusInput)))
        + Math.max(softplusInput, 0);
      const softplusDerivative = 1 / (1 + Math.exp(-softplusInput));
      const negativeA = -Math.exp(aLog[head]);
      const common = gradLogDecay[scalarIndex] * negativeA * softplusDerivative;
      gradA[scalarIndex] = common;
      gradB[scalarIndex] = gradBeta[scalarIndex] * betaValue * (1 - betaValue);
      gradALog[head] += gradLogDecay[scalarIndex] * negativeA * softplus;
      gradDtBias[head] += common;
    }
  }
  return { mixed: gradMixed, a: gradA, b: gradB, aLog: gradALog, dtBias: gradDtBias };
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

function resolveCoreOptions(options) {
  const preparation = resolvePrepareDimensions(options);
  const kernelSize = requirePositiveInteger(options?.kernelSize, 'kernelSize');
  const checkpointInterval = requirePositiveInteger(
    options?.checkpointInterval,
    'checkpointInterval'
  );
  const queryScale = Number(options?.queryScale);
  const rmsEps = Number(options?.rmsEps);
  if (!Number.isFinite(queryScale)) {
    throw new Error('queryScale must be finite.');
  }
  if (!Number.isFinite(rmsEps) || rmsEps <= 0) {
    throw new Error('rmsEps must be finite and positive.');
  }
  return { ...preparation, kernelSize, checkpointInterval, queryScale, rmsEps };
}

export function qwenLinearAttentionCoreForward(inputs, options) {
  const dims = resolveCoreOptions(options);
  const qkv = requireArrayLength(inputs?.qkv, dims.numTokens * dims.convSize, 'qkv');
  const z = requireArrayLength(
    inputs?.z,
    dims.numTokens * dims.numValueHeads * dims.valueDim,
    'z'
  );
  const convWeight = requireArrayLength(
    inputs?.convWeight,
    dims.convSize * dims.kernelSize,
    'convWeight'
  );
  const normWeight = requireArrayLength(inputs?.normWeight, dims.valueDim, 'normWeight');
  const stateElements = dims.numValueHeads * dims.keyDim * dims.valueDim;
  const initialState = requireArrayLength(inputs?.initialState, stateElements, 'initialState');
  const convolution = causalConvSiluForward(qkv, convWeight, {
    numTokens: dims.numTokens,
    channels: dims.convSize,
    kernelSize: dims.kernelSize,
  });
  const preparation = qwenLinearAttentionPrepareForward({
    mixed: convolution.output,
    a: inputs.a,
    b: inputs.b,
    aLog: inputs.aLog,
    dtBias: inputs.dtBias,
  }, dims);
  const recurrenceInputs = {
    query: preparation.query,
    key: preparation.key,
    value: preparation.value,
    logDecay: preparation.logDecay,
    beta: preparation.beta,
    initialState,
  };
  const recurrence = gatedDeltaRecurrentCheckpointedForward(recurrenceInputs, {
    numTokens: dims.numTokens,
    numHeads: dims.numValueHeads,
    keyDim: dims.keyDim,
    valueDim: dims.valueDim,
    queryScale: dims.queryScale,
    checkpointInterval: dims.checkpointInterval,
  });
  const normalization = gatedRmsNormForward(
    recurrence.output,
    z,
    normWeight,
    {
      rows: dims.numTokens * dims.numValueHeads,
      width: dims.valueDim,
      eps: dims.rmsEps,
    }
  );
  return {
    output: normalization.output,
    finalState: recurrence.finalState,
    cache: { dims, convolution, preparation, recurrence, normalization },
  };
}

export function qwenLinearAttentionCoreBackward(inputs, gradOutput, cache, options) {
  const dims = resolveCoreOptions(options);
  for (const key of Object.keys(dims)) {
    if (cache?.dims?.[key] !== dims[key]) {
      throw new Error(`linear-attention core cache mismatch for ${key}.`);
    }
  }
  requireArrayLength(
    gradOutput,
    dims.numTokens * dims.numValueHeads * dims.valueDim,
    'gradOutput'
  );
  const normalizationGradients = gatedRmsNormBackward(
    cache.recurrence.output,
    inputs.z,
    inputs.normWeight,
    gradOutput,
    cache.normalization.cache,
    {
      rows: dims.numTokens * dims.numValueHeads,
      width: dims.valueDim,
      eps: dims.rmsEps,
    }
  );
  const recurrenceInputs = {
    query: cache.preparation.query,
    key: cache.preparation.key,
    value: cache.preparation.value,
    logDecay: cache.preparation.logDecay,
    beta: cache.preparation.beta,
    initialState: inputs.initialState,
  };
  if (inputs.gradFinalState != null) {
    recurrenceInputs.gradFinalState = inputs.gradFinalState;
  }
  const recurrenceGradients = gatedDeltaRecurrentCheckpointedBackward(
    recurrenceInputs,
    normalizationGradients.input,
    cache.recurrence.cache,
    {
      numTokens: dims.numTokens,
      numHeads: dims.numValueHeads,
      keyDim: dims.keyDim,
      valueDim: dims.valueDim,
      queryScale: dims.queryScale,
      checkpointInterval: dims.checkpointInterval,
    }
  );
  const preparationGradients = qwenLinearAttentionPrepareBackward({
    mixed: cache.convolution.output,
    a: inputs.a,
    b: inputs.b,
    aLog: inputs.aLog,
    dtBias: inputs.dtBias,
  }, recurrenceGradients, cache.preparation.cache, dims);
  const convolutionGradients = causalConvSiluBackward(
    inputs.qkv,
    inputs.convWeight,
    preparationGradients.mixed,
    cache.convolution.cache,
    {
      numTokens: dims.numTokens,
      channels: dims.convSize,
      kernelSize: dims.kernelSize,
    }
  );
  return {
    qkv: convolutionGradients.input,
    z: normalizationGradients.gate,
    a: preparationGradients.a,
    b: preparationGradients.b,
    initialState: recurrenceGradients.initialState,
    convWeight: convolutionGradients.weight,
    normWeight: normalizationGradients.weight,
    aLog: preparationGradients.aLog,
    dtBias: preparationGradients.dtBias,
  };
}
