import { sha256BytesHex } from '../../src/utils/sha256.js';

function requireObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value;
}

function requireFiniteArray(value, label) {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error(`${label} must be a non-empty array.`);
  }
  return value.map((entry, index) => {
    const number = Number(entry);
    if (!Number.isFinite(number)) {
      throw new Error(`${label}[${index}] must be finite.`);
    }
    return number;
  });
}

function requireIntegerArray(value, label) {
  const entries = requireFiniteArray(value, label);
  for (let index = 0; index < entries.length; index += 1) {
    if (!Number.isInteger(entries[index]) || entries[index] < 0) {
      throw new Error(`${label}[${index}] must be a non-negative integer.`);
    }
  }
  return entries;
}

function compareExact(actual, expected, label) {
  const mismatches = [];
  const length = Math.max(actual.length, expected.length);
  for (let index = 0; index < length; index += 1) {
    if (actual[index] !== expected[index]) {
      mismatches.push({ index, expected: expected[index] ?? null, actual: actual[index] ?? null });
    }
  }
  return {
    id: label,
    passed: mismatches.length === 0,
    expectedCount: expected.length,
    actualCount: actual.length,
    mismatchCount: mismatches.length,
    mismatches: mismatches.slice(0, 16),
  };
}

function compareIndexed(actual, stride, probes, tolerance, label) {
  const errors = [];
  for (const probe of probes) {
    const position = probe.position ?? 0;
    const indices = requireIntegerArray(probe.indices, `${label}.indices`);
    const values = requireFiniteArray(probe.values, `${label}.values`);
    if (indices.length !== values.length) {
      throw new Error(`${label} indices and values must have equal length.`);
    }
    for (let index = 0; index < indices.length; index += 1) {
      const flatIndex = position * stride + indices[index];
      const actualValue = Number(actual[flatIndex]);
      const expectedValue = values[index];
      const absoluteError = Math.abs(actualValue - expectedValue);
      errors.push({ position, index: indices[index], expected: expectedValue, actual: actualValue, absoluteError });
    }
  }
  const maxAbsoluteError = errors.reduce((maximum, entry) => Math.max(maximum, entry.absoluteError), 0);
  const meanAbsoluteError = errors.reduce((sum, entry) => sum + entry.absoluteError, 0) / errors.length;
  return {
    id: label,
    passed: errors.every((entry) => Number.isFinite(entry.actual) && entry.absoluteError <= tolerance),
    sampleCount: errors.length,
    tolerance,
    maxAbsoluteError,
    meanAbsoluteError,
    failures: errors.filter((entry) => !Number.isFinite(entry.actual) || entry.absoluteError > tolerance).slice(0, 16),
  };
}

function assertFiniteOutput(values, label) {
  if (!ArrayBuffer.isView(values)) {
    throw new Error(`${label} must be a typed array.`);
  }
  let nonFiniteCount = 0;
  for (const value of values) {
    if (!Number.isFinite(value)) nonFiniteCount += 1;
  }
  return {
    id: `${label}.finite`,
    passed: nonFiniteCount === 0,
    valueCount: values.length,
    nonFiniteCount,
  };
}

function outputDigest(values) {
  if (values == null) return null;
  const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  return `sha256:${sha256BytesHex(bytes)}`;
}

function compareOutputs(actual, expected, label, tolerance = 0) {
  if (!ArrayBuffer.isView(actual) || !ArrayBuffer.isView(expected)) {
    throw new Error(`${label} outputs must be typed arrays.`);
  }
  const length = Math.max(actual.length, expected.length);
  let mismatchCount = 0;
  let maxAbsoluteDelta = 0;
  for (let index = 0; index < length; index += 1) {
    const delta = Math.abs(Number(actual[index]) - Number(expected[index]));
    if (!Number.isFinite(delta) || delta > tolerance) mismatchCount += 1;
    if (Number.isFinite(delta)) maxAbsoluteDelta = Math.max(maxAbsoluteDelta, delta);
  }
  return {
    id: label,
    passed: actual.length === expected.length && mismatchCount === 0,
    tolerance,
    expectedCount: expected.length,
    actualCount: actual.length,
    mismatchCount,
    maxAbsoluteDelta,
  };
}

function changedOutput(actual, baseline, label) {
  const comparison = compareOutputs(actual, baseline, label, 0);
  return {
    ...comparison,
    passed: actual.length === baseline.length && comparison.mismatchCount > 0,
    baselineDigest: outputDigest(baseline),
    actualDigest: outputDigest(actual),
  };
}

function deterministicValues(length, scale, offset) {
  return Array.from({ length }, (_, index) => {
    const centered = ((index + offset) % 17) - 8;
    return Math.fround((centered / 8) * scale);
  });
}

export function createSyntheticSequenceLoRAManifest(manifest, options = {}) {
  const modelId = String(manifest?.modelId || '').trim();
  const hiddenSize = Number(manifest?.architecture?.hiddenSize);
  const numAttentionHeads = Number(manifest?.architecture?.numAttentionHeads);
  const headDim = Number(manifest?.architecture?.headDim);
  const layerIndex = options.layerIndex ?? 0;
  if (!modelId) throw new Error('Synthetic sequence LoRA requires manifest.modelId.');
  if (!Number.isInteger(hiddenSize) || hiddenSize < 1) {
    throw new Error('Synthetic sequence LoRA requires architecture.hiddenSize.');
  }
  if (!Number.isInteger(numAttentionHeads) || !Number.isInteger(headDim)) {
    throw new Error('Synthetic sequence LoRA requires attention head dimensions.');
  }
  if (!Number.isInteger(layerIndex) || layerIndex < 0) {
    throw new Error('Synthetic sequence LoRA layerIndex must be a non-negative integer.');
  }
  const outputSize = numAttentionHeads * headDim;
  const name = `${modelId}-synthetic-q-proj-qualification`;
  return {
    id: name,
    name,
    version: '1.0.0',
    baseModel: options.baseModel ?? modelId,
    rank: 1,
    alpha: 1,
    targetModules: ['q_proj'],
    tensors: [
      {
        name: `layers.${layerIndex}.q_proj.lora_a`,
        shape: [1, hiddenSize],
        dtype: 'f32',
        data: deterministicValues(hiddenSize, 0.02, 0),
      },
      {
        name: `layers.${layerIndex}.q_proj.lora_b`,
        shape: [outputSize, 1],
        dtype: 'f32',
        data: deterministicValues(outputSize, 0.02, 5),
      },
    ],
  };
}

export function evaluateSequenceLoRAQualification({
  baseResult,
  adaptedResult,
  restoredResult,
  expectedAdapterName,
  activeAdapterName,
  unloadedAdapterName,
  wrongBaseError,
  invalidLayerError,
}) {
  const outputNames = ['pooledEmbedding', 'tokenEmbeddings'];
  if (baseResult.logits != null) outputNames.push('logits');
  const checks = [
    {
      id: 'lora.wrong-base.rejected',
      passed: /targets base model/u.test(String(wrongBaseError || '')),
      error: wrongBaseError || null,
    },
    {
      id: 'lora.invalid-layer.rejected',
      passed: /targets layer/u.test(String(invalidLayerError || '')),
      error: invalidLayerError || null,
    },
    {
      id: 'lora.activation.active',
      passed: activeAdapterName === expectedAdapterName,
      expectedAdapterName,
      activeAdapterName,
    },
    {
      id: 'lora.unload.inactive',
      passed: unloadedAdapterName == null,
      unloadedAdapterName,
    },
  ];
  for (const name of outputNames) {
    checks.push(assertFiniteOutput(adaptedResult[name], `lora.adapted.${name}`));
    checks.push(changedOutput(adaptedResult[name], baseResult[name], `lora.changed.${name}`));
    checks.push(compareOutputs(restoredResult[name], baseResult[name], `lora.restored.${name}`, 0));
  }
  return {
    passed: checks.every((check) => check.passed),
    checks,
    outputDigests: Object.fromEntries(outputNames.map((name) => [name, {
      base: outputDigest(baseResult[name]),
      adapted: outputDigest(adaptedResult[name]),
      restored: outputDigest(restoredResult[name]),
    }])),
  };
}

function argmaxRows(values, rows, columns) {
  const output = [];
  for (let row = 0; row < rows; row += 1) {
    let best = 0;
    for (let column = 1; column < columns; column += 1) {
      if (values[row * columns + column] > values[row * columns + best]) best = column;
    }
    output.push(best);
  }
  return output;
}

export function validateSequenceReference(reference) {
  requireObject(reference, 'reference');
  if (reference.schema !== 'doppler.sequenceModelReference.v1') {
    throw new Error(`Unsupported sequence reference schema "${reference.schema}".`);
  }
  if (typeof reference.modelId !== 'string' || reference.modelId.length === 0) {
    throw new Error('reference.modelId is required.');
  }
  requireObject(reference.source, 'reference.source');
  requireObject(reference.input, 'reference.input');
  requireObject(reference.probes, 'reference.probes');
  requireObject(reference.tolerances, 'reference.tolerances');
  if (typeof reference.input.sequence !== 'string' || reference.input.sequence.length === 0) {
    throw new Error('reference.input.sequence is required.');
  }
  requireIntegerArray(reference.input.tokenIds, 'reference.input.tokenIds');
  const expectsLogits = reference.outputs?.logits !== false;
  if (expectsLogits) {
    if (!Array.isArray(reference.probes.logits) || reference.probes.logits.length === 0) {
      throw new Error('reference.probes.logits is required when outputs.logits is not false.');
    }
    requireIntegerArray(reference.probes.argmaxTokenIds, 'reference.probes.argmaxTokenIds');
    if (!Number.isFinite(reference.tolerances.logitMaxAbs)) {
      throw new Error('reference.tolerances.logitMaxAbs is required when outputs.logits is not false.');
    }
  }
  return reference;
}

export function evaluateSequenceReference({ manifest, result, reference }) {
  validateSequenceReference(reference);
  const checks = [];
  checks.push({
    id: 'model.identity',
    passed: manifest?.modelId === reference.modelId
      && manifest?.artifactIdentity?.sourceCheckpointId === reference.source.checkpointId,
    expectedModelId: reference.modelId,
    actualModelId: manifest?.modelId ?? null,
    expectedCheckpointId: reference.source.checkpointId ?? null,
    actualCheckpointId: manifest?.artifactIdentity?.sourceCheckpointId ?? null,
  });
  checks.push({
    id: 'sequence.contract',
    passed: manifest?.inference?.supportsSequence === true
      && manifest?.inference?.sequence?.alphabet === reference.input.alphabet,
    expectedAlphabet: reference.input.alphabet,
    actualAlphabet: manifest?.inference?.sequence?.alphabet ?? null,
  });
  checks.push(compareExact(result.tokens, reference.input.tokenIds, 'tokenizer.ids'));
  checks.push(assertFiniteOutput(result.pooledEmbedding, 'pooledEmbedding'));
  checks.push(assertFiniteOutput(result.tokenEmbeddings, 'tokenEmbeddings'));
  const expectsLogits = reference.outputs?.logits !== false;
  if (expectsLogits) {
    checks.push(assertFiniteOutput(result.logits, 'logits'));
  } else {
    checks.push({
      id: 'logits.not-requested',
      passed: result.logits == null,
      actual: result.logits == null ? null : 'present',
    });
  }

  const pooledProbe = requireObject(reference.probes.pooledEmbedding, 'reference.probes.pooledEmbedding');
  checks.push(compareIndexed(
    result.pooledEmbedding,
    result.embeddingDim,
    [{ position: 0, indices: pooledProbe.indices, values: pooledProbe.values }],
    reference.tolerances.pooledEmbeddingMaxAbs,
    'pooledEmbedding.parity'
  ));
  checks.push(compareIndexed(
    result.tokenEmbeddings,
    result.embeddingDim,
    reference.probes.tokenEmbeddings,
    reference.tolerances.tokenEmbeddingMaxAbs,
    'tokenEmbeddings.parity'
  ));
  if (expectsLogits) {
    checks.push(compareIndexed(
      result.logits,
      result.vocabSize,
      reference.probes.logits,
      reference.tolerances.logitMaxAbs,
      'logits.parity'
    ));
    checks.push(compareExact(
      argmaxRows(result.logits, result.tokens.length, result.vocabSize),
      reference.probes.argmaxTokenIds,
      'logits.argmax'
    ));
  }

  return {
    passed: checks.every((check) => check.passed),
    checks,
    outputDigests: {
      pooledEmbedding: outputDigest(result.pooledEmbedding),
      tokenEmbeddings: outputDigest(result.tokenEmbeddings),
      logits: expectsLogits ? outputDigest(result.logits) : null,
    },
  };
}
