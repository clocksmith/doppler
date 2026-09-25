
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

/** @type {import('./layer-partition-contract.js').createLayerPartitionPlan} */
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

  /** @type {import('./layer-partition-contract.js').LayerPartition} */
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

  /** @type {import('./layer-partition-contract.js').LayerPartition} */
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

/** @type {import('./layer-partition-contract.js').validateActivationTensorShape} */
export function validateActivationTensorShape({ shape, dtype, byteLength = null, hiddenSize = null }) {
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
  if (!Number.isSafeInteger(actualHidden) || actualHidden <= 0) {
    throw new RangeError('Activation hidden dimension must be a positive integer.');
  }
  if (hiddenSize != null && actualHidden !== hiddenSize) {
    throw new RangeError(`Activation hiddenSize mismatch: expected ${hiddenSize}, got ${actualHidden}`);
  }
  const bytesPerElem = DTYPE_BYTES[dtype];
  if (!bytesPerElem) {
    throw new TypeError(`Unsupported activation dtype: ${dtype}`);
  }
  const expectedBytes = batchSize * seqLen * actualHidden * bytesPerElem;
  if (!Number.isSafeInteger(expectedBytes)) throw new RangeError('Activation byte size exceeds safe integer range.');
  if (byteLength != null && byteLength !== expectedBytes) {
    throw new RangeError(`Activation byteLength mismatch: expected ${expectedBytes} bytes, got ${byteLength}`);
  }
  return { batchSize, seqLen, hiddenSize: actualHidden, expectedBytes };
}

/** @type {import('./layer-partition-contract.js').serializeActivationFrame} */
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
    byteLength: data.byteLength
  });

  let buffer;
  if (data instanceof ArrayBuffer) {
    buffer = data;
  } else if (ArrayBuffer.isView(data)) {
    buffer = new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice().buffer;
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

/** @type {import('./layer-partition-contract.js').deserializeActivationFrame} */
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

/** @type {import('./layer-partition-contract.js').createPartitionContinuation} */
export function createPartitionContinuation({
  partitionIndex,
  totalLayers,
  layerRange,
  seqOffset = 0
}) {
  let offset = seqOffset;
  const kvCache = new Map();

  /** @type {import('./layer-partition-contract.js').PartitionContinuation} */
  const continuation = {
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
  };
  return Object.freeze(continuation);
}

/** @type {import('./layer-partition-contract.js').comparePartitionExecution} */
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

  if (!Number.isFinite(tolerance) || tolerance < 0 || !Number.isFinite(minCosineSimilarity)
    || minCosineSimilarity < -1 || minCosineSimilarity > 1) {
    throw new RangeError('Finite comparison thresholds in range are required.');
  }
  const length = splitOutput.length;
  if (length === 0) throw new RangeError('Nonempty comparison outputs are required.');
  let maxDiff = 0;
  let sumSquaredDiff = 0;
  let dotProduct = 0;
  let normSplit = 0;
  let normRef = 0;

  for (let i = 0; i < length; i++) {
    const a = splitOutput[i];
    const b = referenceOutput[i];
    if (!Number.isFinite(a) || !Number.isFinite(b)) throw new RangeError('Finite comparison outputs are required.');
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
