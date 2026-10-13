import { computeCanonicalSha256 } from '../../../formats/canonical-hash.js';

export const LAYER_PARTITION_SCHEMA = 'doppler.layer-partition-contract/v1';
export const ACTIVATION_TENSOR_SCHEMA = 'doppler.activation-tensor/v1';
export const ORDERED_LAYER_PARTITION_SCHEMA = 'doppler.layer-partition-contract/v3';
export const PAIRED_LAYER_PARTITION_SCHEMA = 'doppler.layer-partition-contract/v2';
export const PAIRED_ACTIVATION_TENSOR_SCHEMA = 'doppler.activation-tensor/v2';
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
  activationDtype = 'f32',
  activationPrecision
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

  const paired = activationPrecision === 'two-f32';
  if (activationPrecision !== undefined && activationPrecision !== activationDtype && !paired) {
    throw new TypeError('Unsupported activation precision.');
  }
  const format = paired ? resolveActivationTensorLayout({ dtype: activationDtype, precision: activationPrecision,
    components: 2, elementStrideBytes: 8 }) : null;
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
      dtype: activationDtype,
      ...(format || {})
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
      dtype: activationDtype,
      ...(format || {})
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
    schema: paired ? PAIRED_LAYER_PARTITION_SCHEMA : LAYER_PARTITION_SCHEMA,
    modelId,
    totalLayers: numLayers,
    hiddenSize,
    vocabSize,
    splitLayer: split,
    activationDtype,
    ...(format ? { activationPrecision: format.precision, activationComponents: format.components,
      activationElementStrideBytes: format.elementStrideBytes } : {}),
    partitions: Object.freeze([group0, group1].map(group => Object.freeze({
      ...group,
      layerRange: Object.freeze(group.layerRange),
      inputContract: Object.freeze(group.inputContract),
      outputContract: Object.freeze(group.outputContract),
    })))
  });
}

/** Ordered model geometry only; resident execution admits supported plan versions separately.
 * @type {import('./layer-partition-contract.js').createOrderedLayerPartitionPlan} */
export function createOrderedLayerPartitionPlan(options) {
  const { splitLayers } = options;
  if (!Array.isArray(splitLayers) || splitLayers.length < 1) {
    throw new TypeError('Ordered partitions require at least one explicit split layer.');
  }
  let previous = 0;
  for (const split of splitLayers) {
    if (!Number.isSafeInteger(split) || split <= previous || split >= options.numLayers) {
      throw new RangeError('Split layers must be strictly increasing interior model boundaries.');
    }
    previous = split;
  }
  const base = createLayerPartitionPlan({ ...options, splitLayer: splitLayers[0] });
  const layout = resolveActivationTensorLayout(base.partitions[0].outputContract);
  const activation = Object.freeze({ ...base.partitions[0].outputContract, ...layout });
  const boundaries = [0, ...splitLayers, base.totalLayers];
  const stageCount = boundaries.length - 1;
  const partitions = Array.from({ length: stageCount }, (_, index) => Object.freeze({
    index,
    layerRange: Object.freeze([boundaries[index], boundaries[index + 1] - 1]),
    layerCount: boundaries[index + 1] - boundaries[index],
    hasEmbedding: index === 0,
    hasLmHead: index === stageCount - 1,
    inputContract: index === 0 ? base.partitions[0].inputContract : activation,
    outputContract: index === stageCount - 1 ? base.partitions[1].outputContract : activation,
  }));
  return Object.freeze({
    schema: ORDERED_LAYER_PARTITION_SCHEMA, modelId: base.modelId, totalLayers: base.totalLayers,
    hiddenSize: base.hiddenSize, vocabSize: base.vocabSize, stageCount,
    splitLayers: Object.freeze([...splitLayers]), activationDtype: base.activationDtype,
    activationPrecision: layout.precision, activationComponents: layout.components,
    activationElementStrideBytes: layout.elementStrideBytes, partitions: Object.freeze(partitions),
  });
}

/** @param {unknown} actual @param {unknown} expected @returns {boolean} */
function matchesPartitionRecord(actual, expected) {
  if (expected === null || typeof expected !== 'object') return actual === expected;
  if (actual === null || typeof actual !== 'object' || Array.isArray(actual) !== Array.isArray(expected)) return false;
  const keys = Object.keys(expected);
  return Object.keys(actual).length === keys.length && keys.every(key => Object.hasOwn(actual, key)
    && matchesPartitionRecord(Reflect.get(actual, key), Reflect.get(expected, key)));
}

/** @type {import('./layer-partition-contract.js').validateOrderedLayerPartitionPlan} */
export function validateOrderedLayerPartitionPlan(manifest, plan) {
  if (plan?.schema !== ORDERED_LAYER_PARTITION_SCHEMA) throw new Error('Ordered partition plan schema mismatch.');
  const architecture = manifest?.architecture;
  if (!architecture || typeof architecture !== 'object' || !('numLayers' in architecture)
    || !('hiddenSize' in architecture) || !('vocabSize' in architecture)) {
    throw new Error('Ordered partition plan requires explicit model architecture.');
  }
  const canonical = createOrderedLayerPartitionPlan({
    modelId: /** @type {string} */ (manifest.modelId),
    numLayers: /** @type {number} */ (architecture.numLayers),
    hiddenSize: /** @type {number} */ (architecture.hiddenSize),
    vocabSize: /** @type {number} */ (architecture.vocabSize),
    splitLayers: plan.splitLayers, activationDtype: plan.activationDtype, activationPrecision: plan.activationPrecision,
  });
  if (!matchesPartitionRecord(plan, canonical)) throw new Error('Ordered partition plan does not match the model manifest.');
  return canonical;
}

/** @type {import('./layer-partition-contract.js').hashLayerPartitionPlan} */
export function hashLayerPartitionPlan(plan) {
  if (plan?.schema === ORDERED_LAYER_PARTITION_SCHEMA) {
    validateOrderedLayerPartitionPlan({ modelId: plan.modelId, architecture: { numLayers: plan.totalLayers,
      hiddenSize: plan.hiddenSize, vocabSize: plan.vocabSize } }, plan);
  } else if (![LAYER_PARTITION_SCHEMA, PAIRED_LAYER_PARTITION_SCHEMA].includes(plan?.schema)) throw new Error('Layer partition plan schema mismatch.');
  return /** @type {`sha256:${string}`} */ (computeCanonicalSha256(plan));
}

/** @type {import('./layer-partition-contract.js').resolveLayerPartition} */
export function resolveLayerPartition(manifest, allocation) {
  if (allocation == null) return null;
  const { plan, index } = allocation;
  if (plan?.schema === ORDERED_LAYER_PARTITION_SCHEMA) {
    const canonical = validateOrderedLayerPartitionPlan(manifest, plan);
    if (!Number.isSafeInteger(index) || index < 0 || index >= canonical.stageCount) {
      throw new Error('Ordered partition allocation index is outside the declared stages.');
    }
    return canonical.partitions[index];
  }
  if (!plan || ![LAYER_PARTITION_SCHEMA, PAIRED_LAYER_PARTITION_SCHEMA].includes(plan.schema) || ![0, 1].includes(index)) {
    throw new Error('Partition allocation requires a Doppler plan and group index.');
  }
  const architecture = manifest?.architecture;
  if (typeof manifest?.modelId !== 'string' || architecture === null || typeof architecture !== 'object'
    || !('numLayers' in architecture) || typeof architecture.numLayers !== 'number'
    || !('hiddenSize' in architecture) || typeof architecture.hiddenSize !== 'number'
    || !('vocabSize' in architecture) || typeof architecture.vocabSize !== 'number') {
    throw new Error('Partition allocation requires explicit model identity and architecture dimensions.');
  }
  const canonical = createLayerPartitionPlan({ modelId: manifest?.modelId,
    numLayers: architecture?.numLayers, hiddenSize: architecture?.hiddenSize,
    vocabSize: architecture?.vocabSize, splitLayer: plan.splitLayer,
    activationDtype: plan.activationDtype, activationPrecision: plan.activationPrecision });
  /** @type {(actual: unknown, expected: unknown) => boolean} */
  const matches = (actual, expected) => {
    if (expected === null || typeof expected !== 'object') return actual === expected;
    if (actual === null || typeof actual !== 'object'
      || Array.isArray(actual) !== Array.isArray(expected)) return false;
    const keys = Object.keys(expected);
    return Object.keys(actual).length === keys.length
      && keys.every(key => Object.hasOwn(actual, key)
        && matches(Reflect.get(actual, key), Reflect.get(expected, key)));
  };
  if (!matches(plan, canonical)) {
    throw new Error('Partition plan does not match the model manifest.');
  }
  return canonical.partitions[index];
}

/** Byte layout only; no tensor arithmetic or precision conversion. */
/** @type {import('./layer-partition-contract.js').resolveActivationTensorLayout} */
export function resolveActivationTensorLayout({ dtype, precision, components, elementStrideBytes }) {
  const scalarBytes = DTYPE_BYTES[dtype];
  if (!scalarBytes) throw new TypeError('Unsupported activation dtype.');
  if (precision === 'two-f32') {
    if (dtype !== 'f32' || components !== 2 || elementStrideBytes !== 8) {
      throw new TypeError('Paired activation requires f32 precision, two components and eight-byte stride.');
    }
    return { precision, components, elementStrideBytes };
  }
  if ((precision !== undefined && precision !== dtype)
    || (components !== undefined && components !== 1)
    || (elementStrideBytes !== undefined && elementStrideBytes !== scalarBytes)) {
    throw new TypeError('Scalar activation precision, components or stride mismatch.');
  }
  return { precision: dtype, components: 1, elementStrideBytes: scalarBytes };
}

/** @type {import('./layer-partition-contract.js').getPartitionActivationContract} */
export function getPartitionActivationContract(plan) {
  const manifest = { modelId: plan.modelId, architecture: {
    numLayers: plan.totalLayers, hiddenSize: plan.hiddenSize, vocabSize: plan.vocabSize,
  } };
  const partition = plan.schema === ORDERED_LAYER_PARTITION_SCHEMA
    ? validateOrderedLayerPartitionPlan(manifest, plan).partitions[0]
    : resolveLayerPartition(manifest, { plan, index: 0 });
  if (!partition) throw new Error('Partition activation contract is missing.');
  const format = resolveActivationTensorLayout(partition.outputContract);
  return Object.freeze({ schema: format.precision === 'two-f32'
    ? PAIRED_ACTIVATION_TENSOR_SCHEMA : ACTIVATION_TENSOR_SCHEMA,
    dtype: plan.activationDtype, hiddenSize: plan.hiddenSize, ...format });
}

/** @type {import('./layer-partition-contract.js').validateActivationFrameForPlan} */
export function validateActivationFrameForPlan(frame, plan) {
  const expected = getPartitionActivationContract(plan);
  if (frame?.schema !== expected.schema || frame.dtype !== expected.dtype) {
    throw new Error('Activation frame does not match the partition representation.');
  }
  const decoded = deserializeActivationFrame(frame);
  validateActivationTensorShape({ ...expected, shape: decoded.shape, byteLength: decoded.tensorData.byteLength });
  return decoded;
}

/** @type {import('./layer-partition-contract.js').validateActivationTensorShape} */
export function validateActivationTensorShape({ shape, dtype, precision, components, elementStrideBytes, byteLength = null, hiddenSize = null }) {
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
  const { elementStrideBytes: bytesPerElem } = resolveActivationTensorLayout({ dtype, precision, components, elementStrideBytes });
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
  precision, components, elementStrideBytes,
  data,
  seqOffset = 0,
  step = 0,
  metadata = {}
}) {
  const { expectedBytes } = validateActivationTensorShape({
    shape,
    dtype,
    precision, components, elementStrideBytes,
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
    schema: precision === 'two-f32' ? PAIRED_ACTIVATION_TENSOR_SCHEMA : ACTIVATION_TENSOR_SCHEMA,
    shape: Object.freeze([...shape]),
    dtype,
    ...(precision === 'two-f32' ? { precision, components, elementStrideBytes } : {}),
    seqOffset,
    step,
    byteLength: buffer.byteLength,
    metadata: Object.freeze({ ...metadata }),
    buffer
  });
}

/** @type {import('./layer-partition-contract.js').deserializeActivationFrame} */
export function deserializeActivationFrame(frame) {
  if (!frame || ![ACTIVATION_TENSOR_SCHEMA, PAIRED_ACTIVATION_TENSOR_SCHEMA].includes(frame.schema)) {
    throw new TypeError(`Invalid activation frame schema: ${frame?.schema}`);
  }
  const { shape, dtype, buffer, seqOffset, step, precision, components, elementStrideBytes } = frame;
  if (frame.schema === PAIRED_ACTIVATION_TENSOR_SCHEMA ? precision !== 'two-f32'
    : precision !== undefined || components !== undefined || elementStrideBytes !== undefined) {
    throw new TypeError('Activation schema and paired precision disagree.');
  }
  if (!(buffer instanceof ArrayBuffer) || frame.byteLength !== buffer.byteLength) {
    throw new RangeError('Activation frame byteLength mismatch.');
  }
  validateActivationTensorShape({ shape, dtype, precision, components, elementStrideBytes, byteLength: buffer.byteLength });

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
    ...(precision === 'two-f32' ? { precision, components, elementStrideBytes } : {}),
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
