/**
 * @fileoverview Doppler public layer partition contract.
 *
 * Defines executable partition shapes, intermediate activation tensor contracts,
 * isolated continuation states, and numerical comparison requirements for split
 * layer execution across cooperating devices.
 */

export const LAYER_PARTITION_SCHEMA = 'doppler.layer-partition-contract/v1';
export const ACTIVATION_TENSOR_SCHEMA = 'doppler.activation-tensor/v1';
export const PARTITION_COMPARISON_SCHEMA = 'doppler.partition-comparison-contract/v1';
export const DEFAULT_NUMERICAL_TOLERANCE = 1e-4;
export const DEFAULT_COSINE_SIMILARITY_MIN = 0.9999;

const DTYPE_BYTES = Object.freeze({
  f32: 4,
  f16: 2,
  i32: 4
});

/**
 * Creates an executable 2-partition layer plan for a transformer model.
 *
 * @param {Object} options
 * @param {string} options.modelId - Model identifier
 * @param {number} options.numLayers - Total number of transformer layers
 * @param {number} options.hiddenSize - Hidden activation dimension D
 * @param {number} options.vocabSize - Vocabulary size V
 * @param {number} [options.splitLayer] - Index where partition 1 starts (default: Math.floor(numLayers / 2))
 * @param {string} [options.activationDtype='f32'] - Dtype for intermediate activation tensor ('f32' | 'f16')
 * @returns {Object} Validated partition plan
 */
export function createLayerPartitionPlan({
  modelId,
  numLayers,
  hiddenSize,
  vocabSize,
  splitLayer = null,
  activationDtype = 'f32'
}) {
  if (!modelId || typeof modelId !== 'string') {
    throw new TypeError('modelId must be a non-empty string');
  }
  if (!Number.isSafeInteger(numLayers) || numLayers < 2) {
    throw new RangeError(`numLayers must be an integer >= 2; got ${numLayers}`);
  }
  if (!Number.isSafeInteger(hiddenSize) || hiddenSize <= 0) {
    throw new RangeError(`hiddenSize must be a positive integer; got ${hiddenSize}`);
  }
  if (!Number.isSafeInteger(vocabSize) || vocabSize <= 0) {
    throw new RangeError(`vocabSize must be a positive integer; got ${vocabSize}`);
  }
  if (!['f32', 'f16'].includes(activationDtype)) {
    throw new TypeError(`activationDtype must be 'f32' or 'f16'; got "${activationDtype}"`);
  }

  const split = splitLayer === null ? Math.floor(numLayers / 2) : splitLayer;
  if (!Number.isSafeInteger(split) || split < 1 || split >= numLayers) {
    throw new RangeError(`splitLayer must be in range [1, ${numLayers - 1}]; got ${split}`);
  }

  const group0 = {
    index: 0,
    layerRange: [0, split - 1],
    layerCount: split,
    hasEmbedding: true,
    hasLmHead: false,
    inputContract: {
      type: 'token-ids',
      rank: 2,
      shapeDescription: '[batchSize, seqLen]',
      dtype: 'i32'
    },
    outputContract: {
      type: 'activation-tensor',
      rank: 3,
      shapeDescription: '[batchSize, seqLen, hiddenSize]',
      hiddenSize,
      dtype: activationDtype
    }
  };

  const group1 = {
    index: 1,
    layerRange: [split, numLayers - 1],
    layerCount: numLayers - split,
    hasEmbedding: false,
    hasLmHead: true,
    inputContract: {
      type: 'activation-tensor',
      rank: 3,
      shapeDescription: '[batchSize, seqLen, hiddenSize]',
      hiddenSize,
      dtype: activationDtype
    },
    outputContract: {
      type: 'logits',
      rank: 3,
      shapeDescription: '[batchSize, seqLen, vocabSize]',
      vocabSize,
      dtype: 'f32'
    }
  };

  return Object.freeze({
    schema: LAYER_PARTITION_SCHEMA,
    modelId,
    totalLayers: numLayers,
    hiddenSize,
    vocabSize,
    splitLayer: split,
    activationDtype,
    partitions: Object.freeze([Object.freeze(group0), Object.freeze(group1)])
  });
}

/**
 * Validates whether an activation tensor shape and byte length match the contract.
 */
export function validateActivationTensorShape({ shape, dtype, byteLength, hiddenSize }) {
  if (!Array.isArray(shape) || shape.length !== 3) {
    throw new TypeError(`Activation shape must be 3D [batchSize, seqLen, hiddenSize]; got ${JSON.stringify(shape)}`);
  }
  const [batchSize, seqLen, actualHidden] = shape;
  if (!Number.isSafeInteger(batchSize) || batchSize <= 0) {
    throw new RangeError(`batchSize must be a positive integer; got ${batchSize}`);
  }
  if (!Number.isSafeInteger(seqLen) || seqLen <= 0) {
    throw new RangeError(`seqLen must be a positive integer; got ${seqLen}`);
  }
  if (hiddenSize != null && actualHidden !== hiddenSize) {
    throw new RangeError(`Activation hiddenSize mismatch: expected ${hiddenSize}, got ${actualHidden}`);
  }
  const bytesPerElem = DTYPE_BYTES[dtype];
  if (!bytesPerElem) {
    throw new TypeError(`Unsupported activation dtype: ${dtype}`);
  }
  const expectedBytes = batchSize * seqLen * actualHidden * bytesPerElem;
  if (byteLength != null && byteLength !== expectedBytes) {
    throw new RangeError(`Activation byteLength mismatch: expected ${expectedBytes} bytes, got ${byteLength}`);
  }
  return { batchSize, seqLen, hiddenSize: actualHidden, expectedBytes };
}

/**
 * Serializes an intermediate activation frame for wire transfer over WebRTC.
 */
export function serializeActivationFrame({
  shape,
  dtype = 'f32',
  data,
  seqOffset = 0,
  step = 0,
  metadata = {}
}) {
  const { expectedBytes } = validateActivationTensorShape({
    shape,
    dtype,
    byteLength: data.byteLength ?? (data.length * DTYPE_BYTES[dtype])
  });

  let buffer;
  if (data instanceof ArrayBuffer) {
    buffer = data;
  } else if (ArrayBuffer.isView(data)) {
    buffer = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  } else {
    throw new TypeError('Activation data must be an ArrayBuffer or ArrayBufferView');
  }

  if (buffer.byteLength !== expectedBytes) {
    throw new RangeError(`Buffer byteLength (${buffer.byteLength}) does not match expected tensor size (${expectedBytes})`);
  }

  return Object.freeze({
    schema: ACTIVATION_TENSOR_SCHEMA,
    shape: Object.freeze([...shape]),
    dtype,
    seqOffset,
    step,
    byteLength: buffer.byteLength,
    metadata: Object.freeze({ ...metadata }),
    buffer
  });
}

/**
 * Deserializes an activation frame received over WebRTC into a typed array.
 */
export function deserializeActivationFrame(frame) {
  if (!frame || frame.schema !== ACTIVATION_TENSOR_SCHEMA) {
    throw new TypeError(`Invalid activation frame schema: ${frame?.schema}`);
  }
  const { shape, dtype, buffer, seqOffset, step } = frame;
  validateActivationTensorShape({ shape, dtype, byteLength: buffer.byteLength });

  let tensorData;
  if (dtype === 'f32') {
    tensorData = new Float32Array(buffer);
  } else if (dtype === 'f16') {
    tensorData = new Uint16Array(buffer);
  } else {
    throw new TypeError(`Unsupported dtype for activation frame: ${dtype}`);
  }

  return {
    shape: [...shape],
    dtype,
    seqOffset: seqOffset ?? 0,
    step: step ?? 0,
    tensorData,
    metadata: frame.metadata || {}
  };
}

/**
 * Creates isolated continuation state for a layer group.
 */
export function createPartitionContinuation({
  partitionIndex,
  totalLayers,
  layerRange,
  seqOffset = 0
}) {
  let offset = seqOffset;
  const kvCache = new Map();

  return Object.freeze({
    partitionIndex,
    layerRange: Object.freeze([...layerRange]),
    getSequenceOffset() {
      return offset;
    },
    advance(steps = 1) {
      offset += steps;
      return offset;
    },
    getLayerKVCache(layerIdx) {
      if (layerIdx < layerRange[0] || layerIdx > layerRange[1]) {
        throw new RangeError(`Layer ${layerIdx} is outside partition ${partitionIndex} range [${layerRange.join(', ')}]`);
      }
      return kvCache.get(layerIdx) || null;
    },
    setLayerKVCache(layerIdx, state) {
      if (layerIdx < layerRange[0] || layerIdx > layerRange[1]) {
        throw new RangeError(`Layer ${layerIdx} is outside partition ${partitionIndex} range [${layerRange.join(', ')}]`);
      }
      kvCache.set(layerIdx, state);
    },
    reset() {
      offset = 0;
      kvCache.clear();
    }
  });
}

/**
 * Compares numerical output from split execution against unsplit reference output.
 *
 * @param {Object} options
 * @param {Float32Array|number[]} options.splitOutput - Logits from split 2-partition execution
 * @param {Float32Array|number[]} options.referenceOutput - Logits from single-device unsplit execution
 * @param {number} [options.tolerance=DEFAULT_NUMERICAL_TOLERANCE] - Max absolute error threshold
 * @param {number} [options.minCosineSimilarity=DEFAULT_COSINE_SIMILARITY_MIN] - Minimum cosine similarity
 * @returns {Object} Comparison assessment
 */
export function comparePartitionExecution({
  splitOutput,
  referenceOutput,
  tolerance = DEFAULT_NUMERICAL_TOLERANCE,
  minCosineSimilarity = DEFAULT_COSINE_SIMILARITY_MIN
}) {
  if (!splitOutput || !referenceOutput) {
    throw new TypeError('splitOutput and referenceOutput are required');
  }
  if (splitOutput.length !== referenceOutput.length) {
    throw new RangeError(
      `Output length mismatch: split length ${splitOutput.length} vs reference length ${referenceOutput.length}`
    );
  }

  const length = splitOutput.length;
  let maxDiff = 0;
  let sumSquaredDiff = 0;
  let dotProduct = 0;
  let normSplit = 0;
  let normRef = 0;

  for (let i = 0; i < length; i++) {
    const a = splitOutput[i];
    const b = referenceOutput[i];
    const diff = Math.abs(a - b);
    if (diff > maxDiff) maxDiff = diff;
    sumSquaredDiff += diff * diff;
    dotProduct += a * b;
    normSplit += a * a;
    normRef += b * b;
  }

  const mse = length > 0 ? sumSquaredDiff / length : 0;
  const denom = Math.sqrt(normSplit) * Math.sqrt(normRef);
  const cosineSimilarity = denom > 0 ? dotProduct / denom : (maxDiff === 0 ? 1 : 0);
  const matches = maxDiff <= tolerance && cosineSimilarity >= minCosineSimilarity;

  return Object.freeze({
    schema: PARTITION_COMPARISON_SCHEMA,
    matches,
    maxDiff,
    mse,
    cosineSimilarity,
    tolerance,
    minCosineSimilarity,
    length
  });
}
