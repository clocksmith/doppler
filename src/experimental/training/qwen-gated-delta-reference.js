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

function resolveDimensions(options) {
  const numTokens = requirePositiveInteger(options?.numTokens, 'numTokens');
  const numHeads = requirePositiveInteger(options?.numHeads, 'numHeads');
  const keyDim = requirePositiveInteger(options?.keyDim, 'keyDim');
  const valueDim = requirePositiveInteger(options?.valueDim, 'valueDim');
  const queryScale = Number(options?.queryScale);
  if (!Number.isFinite(queryScale)) {
    throw new Error('queryScale must be finite.');
  }
  return { numTokens, numHeads, keyDim, valueDim, queryScale };
}

function vectorIndex(token, head, dim, numHeads, width) {
  return ((token * numHeads) + head) * width + dim;
}

function stateIndex(token, head, key, value, numHeads, keyDim, valueDim) {
  return ((((token * numHeads) + head) * keyDim) + key) * valueDim + value;
}

export function gatedDeltaRecurrentForward(inputs, options) {
  const dims = resolveDimensions(options);
  const { numTokens, numHeads, keyDim, valueDim, queryScale } = dims;
  const query = requireArrayLength(inputs?.query, numTokens * numHeads * keyDim, 'query');
  const key = requireArrayLength(inputs?.key, numTokens * numHeads * keyDim, 'key');
  const value = requireArrayLength(inputs?.value, numTokens * numHeads * valueDim, 'value');
  const logDecay = requireArrayLength(inputs?.logDecay, numTokens * numHeads, 'logDecay');
  const beta = requireArrayLength(inputs?.beta, numTokens * numHeads, 'beta');
  const initialState = requireArrayLength(
    inputs?.initialState,
    numHeads * keyDim * valueDim,
    'initialState'
  );
  const stateStride = numHeads * keyDim * valueDim;
  const states = new Float32Array((numTokens + 1) * stateStride);
  states.set(initialState, 0);
  const output = new Float32Array(numTokens * numHeads * valueDim);

  for (let token = 0; token < numTokens; token += 1) {
    for (let head = 0; head < numHeads; head += 1) {
      const scalarIndex = (token * numHeads) + head;
      const decay = Math.exp(logDecay[scalarIndex]);
      const betaValue = beta[scalarIndex];
      for (let valueIndex = 0; valueIndex < valueDim; valueIndex += 1) {
        let memoryValue = 0;
        for (let keyIndex = 0; keyIndex < keyDim; keyIndex += 1) {
          const previousIndex = stateIndex(
            token,
            head,
            keyIndex,
            valueIndex,
            numHeads,
            keyDim,
            valueDim
          );
          const keyValue = key[vectorIndex(token, head, keyIndex, numHeads, keyDim)];
          memoryValue += states[previousIndex] * decay * keyValue;
        }
        const valueOffset = vectorIndex(token, head, valueIndex, numHeads, valueDim);
        const delta = (value[valueOffset] - memoryValue) * betaValue;
        let outputValue = 0;
        for (let keyIndex = 0; keyIndex < keyDim; keyIndex += 1) {
          const previousIndex = stateIndex(
            token,
            head,
            keyIndex,
            valueIndex,
            numHeads,
            keyDim,
            valueDim
          );
          const nextIndex = stateIndex(
            token + 1,
            head,
            keyIndex,
            valueIndex,
            numHeads,
            keyDim,
            valueDim
          );
          const keyValue = key[vectorIndex(token, head, keyIndex, numHeads, keyDim)];
          const nextState = (states[previousIndex] * decay) + (keyValue * delta);
          states[nextIndex] = nextState;
          outputValue += nextState
            * query[vectorIndex(token, head, keyIndex, numHeads, keyDim)]
            * queryScale;
        }
        output[valueOffset] = outputValue;
      }
    }
  }

  return {
    output,
    finalState: states.slice(numTokens * stateStride),
    cache: { states, dims },
  };
}

export function gatedDeltaRecurrentBackward(inputs, gradOutput, cache, options) {
  const dims = resolveDimensions(options);
  const cachedDims = cache?.dims;
  for (const key of ['numTokens', 'numHeads', 'keyDim', 'valueDim', 'queryScale']) {
    if (cachedDims?.[key] !== dims[key]) {
      throw new Error(`gated-delta backward cache mismatch for ${key}.`);
    }
  }
  const { numTokens, numHeads, keyDim, valueDim, queryScale } = dims;
  const query = requireArrayLength(inputs?.query, numTokens * numHeads * keyDim, 'query');
  const key = requireArrayLength(inputs?.key, numTokens * numHeads * keyDim, 'key');
  const value = requireArrayLength(inputs?.value, numTokens * numHeads * valueDim, 'value');
  const logDecay = requireArrayLength(inputs?.logDecay, numTokens * numHeads, 'logDecay');
  const beta = requireArrayLength(inputs?.beta, numTokens * numHeads, 'beta');
  requireArrayLength(inputs?.initialState, numHeads * keyDim * valueDim, 'initialState');
  requireArrayLength(gradOutput, numTokens * numHeads * valueDim, 'gradOutput');
  const states = requireArrayLength(
    cache?.states,
    (numTokens + 1) * numHeads * keyDim * valueDim,
    'cache.states'
  );
  const gradQuery = new Float32Array(query.length);
  const gradKey = new Float32Array(key.length);
  const gradValue = new Float32Array(value.length);
  const gradLogDecay = new Float32Array(logDecay.length);
  const gradBeta = new Float32Array(beta.length);
  let gradStateNext = inputs?.gradFinalState == null
    ? new Float32Array(numHeads * keyDim * valueDim)
    : new Float32Array(requireArrayLength(
      inputs.gradFinalState,
      numHeads * keyDim * valueDim,
      'gradFinalState'
    ));

  for (let token = numTokens - 1; token >= 0; token -= 1) {
    const gradStatePrevious = new Float32Array(gradStateNext.length);
    for (let head = 0; head < numHeads; head += 1) {
      const scalarIndex = (token * numHeads) + head;
      const decay = Math.exp(logDecay[scalarIndex]);
      const betaValue = beta[scalarIndex];
      const memory = new Float32Array(valueDim);
      const delta = new Float32Array(valueDim);
      const gradMemory = new Float32Array(valueDim);
      const gradDelta = new Float32Array(valueDim);

      for (let valueIndex = 0; valueIndex < valueDim; valueIndex += 1) {
        for (let keyIndex = 0; keyIndex < keyDim; keyIndex += 1) {
          const previousIndex = stateIndex(
            token,
            head,
            keyIndex,
            valueIndex,
            numHeads,
            keyDim,
            valueDim
          );
          memory[valueIndex] += states[previousIndex]
            * decay
            * key[vectorIndex(token, head, keyIndex, numHeads, keyDim)];
        }
        const valueOffset = vectorIndex(token, head, valueIndex, numHeads, valueDim);
        delta[valueIndex] = (value[valueOffset] - memory[valueIndex]) * betaValue;
      }

      for (let valueIndex = 0; valueIndex < valueDim; valueIndex += 1) {
        const outputOffset = vectorIndex(token, head, valueIndex, numHeads, valueDim);
        const outputGradient = gradOutput[outputOffset];
        for (let keyIndex = 0; keyIndex < keyDim; keyIndex += 1) {
          const stateOffset = (((head * keyDim) + keyIndex) * valueDim) + valueIndex;
          const nextIndex = stateIndex(
            token + 1,
            head,
            keyIndex,
            valueIndex,
            numHeads,
            keyDim,
            valueDim
          );
          const queryOffset = vectorIndex(token, head, keyIndex, numHeads, keyDim);
          gradStateNext[stateOffset] += outputGradient * query[queryOffset] * queryScale;
          gradQuery[queryOffset] += outputGradient * states[nextIndex] * queryScale;
        }
      }

      for (let valueIndex = 0; valueIndex < valueDim; valueIndex += 1) {
        const valueOffset = vectorIndex(token, head, valueIndex, numHeads, valueDim);
        for (let keyIndex = 0; keyIndex < keyDim; keyIndex += 1) {
          const stateOffset = (((head * keyDim) + keyIndex) * valueDim) + valueIndex;
          const keyOffset = vectorIndex(token, head, keyIndex, numHeads, keyDim);
          gradKey[keyOffset] += gradStateNext[stateOffset] * delta[valueIndex];
          gradDelta[valueIndex] += gradStateNext[stateOffset] * key[keyOffset];
        }
        gradValue[valueOffset] += gradDelta[valueIndex] * betaValue;
        gradMemory[valueIndex] -= gradDelta[valueIndex] * betaValue;
        gradBeta[scalarIndex] += gradDelta[valueIndex] * (value[valueOffset] - memory[valueIndex]);
      }

      for (let valueIndex = 0; valueIndex < valueDim; valueIndex += 1) {
        for (let keyIndex = 0; keyIndex < keyDim; keyIndex += 1) {
          const stateOffset = (((head * keyDim) + keyIndex) * valueDim) + valueIndex;
          const previousIndex = stateIndex(
            token,
            head,
            keyIndex,
            valueIndex,
            numHeads,
            keyDim,
            valueDim
          );
          const keyOffset = vectorIndex(token, head, keyIndex, numHeads, keyDim);
          const decayedState = states[previousIndex] * decay;
          gradStateNext[stateOffset] += gradMemory[valueIndex] * key[keyOffset];
          gradKey[keyOffset] += gradMemory[valueIndex] * decayedState;
          gradStatePrevious[stateOffset] += gradStateNext[stateOffset] * decay;
          gradLogDecay[scalarIndex] += gradStateNext[stateOffset] * decayedState;
        }
      }
    }
    gradStateNext = gradStatePrevious;
  }

  return {
    query: gradQuery,
    key: gradKey,
    value: gradValue,
    logDecay: gradLogDecay,
    beta: gradBeta,
    initialState: gradStateNext,
  };
}

function sliceTokenRows(values, startToken, tokenCount, rowWidth) {
  const start = startToken * rowWidth;
  return values.slice(start, start + (tokenCount * rowWidth));
}

function sliceBlockInputs(inputs, startToken, tokenCount, dims, initialState) {
  return {
    query: sliceTokenRows(inputs.query, startToken, tokenCount, dims.numHeads * dims.keyDim),
    key: sliceTokenRows(inputs.key, startToken, tokenCount, dims.numHeads * dims.keyDim),
    value: sliceTokenRows(inputs.value, startToken, tokenCount, dims.numHeads * dims.valueDim),
    logDecay: sliceTokenRows(inputs.logDecay, startToken, tokenCount, dims.numHeads),
    beta: sliceTokenRows(inputs.beta, startToken, tokenCount, dims.numHeads),
    initialState: new Float32Array(initialState),
  };
}

function resolveCheckpointInterval(options, numTokens) {
  const interval = Math.floor(Number(options?.checkpointInterval));
  if (!Number.isInteger(interval) || interval < 1) {
    throw new Error('checkpointInterval must be a positive integer.');
  }
  return Math.min(interval, numTokens);
}

export function estimateGatedDeltaCheckpointElements(options) {
  const dims = resolveDimensions(options);
  const checkpointInterval = resolveCheckpointInterval(options, dims.numTokens);
  const stateElements = dims.numHeads * dims.keyDim * dims.valueDim;
  const blockCount = Math.ceil(dims.numTokens / checkpointInterval);
  return {
    checkpointInterval,
    blockCount,
    stateElements,
    fullHistoryElements: (dims.numTokens + 1) * stateElements,
    storedCheckpointElements: (blockCount + 1) * stateElements,
    peakRecomputedBlockElements: (checkpointInterval + 1) * stateElements,
    peakBackwardStateElements: (blockCount + checkpointInterval + 2) * stateElements,
  };
}

export function gatedDeltaRecurrentCheckpointedForward(inputs, options) {
  const dims = resolveDimensions(options);
  const checkpointInterval = resolveCheckpointInterval(options, dims.numTokens);
  requireArrayLength(inputs?.query, dims.numTokens * dims.numHeads * dims.keyDim, 'query');
  requireArrayLength(inputs?.key, dims.numTokens * dims.numHeads * dims.keyDim, 'key');
  requireArrayLength(inputs?.value, dims.numTokens * dims.numHeads * dims.valueDim, 'value');
  requireArrayLength(inputs?.logDecay, dims.numTokens * dims.numHeads, 'logDecay');
  requireArrayLength(inputs?.beta, dims.numTokens * dims.numHeads, 'beta');
  const stateElements = dims.numHeads * dims.keyDim * dims.valueDim;
  let currentState = new Float32Array(requireArrayLength(
    inputs?.initialState,
    stateElements,
    'initialState'
  ));
  const blockCount = Math.ceil(dims.numTokens / checkpointInterval);
  const checkpoints = new Float32Array((blockCount + 1) * stateElements);
  const checkpointTokens = new Uint32Array(blockCount + 1);
  const output = new Float32Array(dims.numTokens * dims.numHeads * dims.valueDim);
  checkpoints.set(currentState, 0);

  for (let block = 0; block < blockCount; block += 1) {
    const startToken = block * checkpointInterval;
    const tokenCount = Math.min(checkpointInterval, dims.numTokens - startToken);
    const blockInputs = sliceBlockInputs(inputs, startToken, tokenCount, dims, currentState);
    const blockOptions = { ...dims, numTokens: tokenCount };
    const blockForward = gatedDeltaRecurrentForward(blockInputs, blockOptions);
    output.set(blockForward.output, startToken * dims.numHeads * dims.valueDim);
    currentState = blockForward.finalState;
    checkpoints.set(currentState, (block + 1) * stateElements);
    checkpointTokens[block + 1] = startToken + tokenCount;
  }

  return {
    output,
    finalState: currentState,
    cache: {
      checkpoints,
      checkpointTokens,
      checkpointInterval,
      blockCount,
      dims,
    },
  };
}

export function gatedDeltaRecurrentCheckpointedBackward(
  inputs,
  gradOutput,
  cache,
  options
) {
  const dims = resolveDimensions(options);
  const checkpointInterval = resolveCheckpointInterval(options, dims.numTokens);
  if (cache?.checkpointInterval !== checkpointInterval || cache?.blockCount !== Math.ceil(dims.numTokens / checkpointInterval)) {
    throw new Error('checkpointed gated-delta backward cache schedule does not match options.');
  }
  const stateElements = dims.numHeads * dims.keyDim * dims.valueDim;
  const checkpoints = requireArrayLength(
    cache?.checkpoints,
    (cache.blockCount + 1) * stateElements,
    'cache.checkpoints'
  );
  requireArrayLength(gradOutput, dims.numTokens * dims.numHeads * dims.valueDim, 'gradOutput');
  const gradients = {
    query: new Float32Array(inputs.query.length),
    key: new Float32Array(inputs.key.length),
    value: new Float32Array(inputs.value.length),
    logDecay: new Float32Array(inputs.logDecay.length),
    beta: new Float32Array(inputs.beta.length),
    initialState: null,
  };
  let gradFinalState = inputs?.gradFinalState == null
    ? new Float32Array(stateElements)
    : new Float32Array(requireArrayLength(inputs.gradFinalState, stateElements, 'gradFinalState'));

  for (let block = cache.blockCount - 1; block >= 0; block -= 1) {
    const startToken = block * checkpointInterval;
    const tokenCount = Math.min(checkpointInterval, dims.numTokens - startToken);
    const checkpointStart = block * stateElements;
    const initialState = checkpoints.slice(checkpointStart, checkpointStart + stateElements);
    const blockInputs = sliceBlockInputs(inputs, startToken, tokenCount, dims, initialState);
    blockInputs.gradFinalState = gradFinalState;
    const blockOptions = { ...dims, numTokens: tokenCount };
    const recomputed = gatedDeltaRecurrentForward(blockInputs, blockOptions);
    const blockGradOutput = sliceTokenRows(
      gradOutput,
      startToken,
      tokenCount,
      dims.numHeads * dims.valueDim
    );
    const blockGradients = gatedDeltaRecurrentBackward(
      blockInputs,
      blockGradOutput,
      recomputed.cache,
      blockOptions
    );
    gradients.query.set(blockGradients.query, startToken * dims.numHeads * dims.keyDim);
    gradients.key.set(blockGradients.key, startToken * dims.numHeads * dims.keyDim);
    gradients.value.set(blockGradients.value, startToken * dims.numHeads * dims.valueDim);
    gradients.logDecay.set(blockGradients.logDecay, startToken * dims.numHeads);
    gradients.beta.set(blockGradients.beta, startToken * dims.numHeads);
    gradFinalState = blockGradients.initialState;
  }
  gradients.initialState = gradFinalState;
  return gradients;
}
