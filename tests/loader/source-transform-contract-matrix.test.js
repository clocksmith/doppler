import assert from 'node:assert/strict';

import { normalizeTensorSourceTransform } from '../../src/formats/rdrr/source-transform-contract.js';
import { getSourceTransformSpec } from '../../src/loader/tensors/source-transform.js';

const validCases = [
  {
    label: 'affine int8',
    location: {
      dtype: 'F16',
      shape: [2, 2],
      sourceTransform: {
        kind: 'affine_dequant',
        scheme: 'per_tensor_affine',
        sourceDtype: 'INT8',
        targetDtype: 'F16',
        scale: 0.25,
        zeroPoint: 1,
      },
    },
    expected: {
      kind: 'affine_dequant',
      sourceByteLength: 4,
      outputByteLength: 8,
    },
  },
  {
    label: 'rowwise int4 qmax_abs',
    location: {
      dtype: 'F16',
      shape: [2, 4],
      sourceTransform: {
        kind: 'litert_rowwise_dequant',
        scheme: 'per_row_affine',
        sourceDtype: 'INT4',
        targetDtype: 'F16',
        storageEncoding: 'offset_binary',
        scaleSemantics: 'qmax_abs',
        scaleDivisor: 8,
        scaleSource: { shard: 1, offset: 0, size: 8 },
        rowSumSource: { shard: 2, offset: 0, size: 8 },
      },
    },
    expected: {
      kind: 'litert_rowwise_dequant',
      sourceByteLength: 4,
      outputByteLength: 16,
    },
  },
  {
    label: 'axis uint8 companion',
    location: {
      dtype: 'F16',
      shape: [2, 4],
      sourceTransform: {
        kind: 'litert_axis_dequant',
        scheme: 'per_axis_affine',
        sourceDtype: 'UINT8',
        targetDtype: 'F16',
        storageEncoding: 'signed',
        scaleSemantics: 'step',
        storageShape: [2, 4],
        quantAxis: 1,
        scaleCompanionDtype: 'UINT8',
        scaleCompanionDequant: {
          scale: 0.01,
          zeroPoint: 0,
        },
        scaleSource: { shard: 1, offset: 0, size: 8 },
      },
    },
    expected: {
      kind: 'litert_axis_dequant',
      sourceByteLength: 8,
      outputByteLength: 16,
      scaleCompanionDtype: 'UINT8',
    },
  },
  {
    label: 'blocked axis int2',
    location: {
      dtype: 'F16',
      shape: [2, 8],
      sourceTransform: {
        kind: 'litert_axis_blocked_dequant',
        scheme: 'per_axis_affine',
        sourceDtype: 'INT2',
        targetDtype: 'F16',
        storageEncoding: 'offset_binary',
        scaleSemantics: 'step',
        storageShape: [2, 2],
        quantAxis: 0,
        storageBlockSize: 4,
        storageLaneOrder: [0, 1, 2, 3],
        scaleSource: { shard: 1, offset: 0, size: 8 },
        sumSource: { shard: 2, offset: 0, size: 8 },
      },
    },
    expected: {
      kind: 'litert_axis_blocked_dequant',
      sourceByteLength: 4,
      outputByteLength: 32,
    },
  },
];

for (const testCase of validCases) {
  const normalized = normalizeTensorSourceTransform(
    testCase.location,
    `matrix/${testCase.label}`
  );
  assert.equal(normalized.kind, testCase.expected.kind);
  if (testCase.expected.scaleCompanionDtype) {
    assert.equal(normalized.scaleCompanionDtype, testCase.expected.scaleCompanionDtype);
  }
  const spec = getSourceTransformSpec(testCase.location, `matrix/${testCase.label}`);
  assert.equal(spec.kind, testCase.expected.kind);
  assert.equal(spec.sourceByteLength, testCase.expected.sourceByteLength);
  assert.equal(spec.outputByteLength, testCase.expected.outputByteLength);
}

assert.throws(
  () => normalizeTensorSourceTransform({
    dtype: 'F16',
    shape: [2, 4],
    sourceTransform: {
      kind: 'litert_axis_dequant',
      scheme: 'per_axis_affine',
      sourceDtype: 'UINT8',
      targetDtype: 'F16',
      storageEncoding: 'signed',
      scaleSemantics: 'step',
      storageShape: [2, 4],
      quantAxis: 1,
      scaleCompanionDtype: 'UINT8',
      scaleSource: { shard: 1, offset: 0, size: 8 },
    },
  }, 'matrix/missing-companion-dequant'),
  /UINT8 scale companion without scaleCompanionDequant metadata/
);

assert.throws(
  () => normalizeTensorSourceTransform({
    dtype: 'F16',
    shape: [2, 4],
    sourceTransform: {
      kind: 'litert_axis_dequant',
      scheme: 'per_axis_affine',
      sourceDtype: 'UINT8',
      targetDtype: 'F16',
      storageEncoding: 'signed',
      scaleSemantics: 'step',
      storageShape: [2, 4],
      quantAxis: 1,
      scaleCompanionDtype: 'F32',
      scaleSource: { shard: 1, offset: 0, size: 8 },
    },
  }, 'matrix/unsupported-companion-dtype'),
  /unsupported scaleCompanionDtype/
);

assert.throws(
  () => normalizeTensorSourceTransform({
    dtype: 'F16',
    shape: [2, 8],
    sourceTransform: {
      kind: 'litert_axis_blocked_dequant',
      scheme: 'per_axis_affine',
      sourceDtype: 'INT2',
      targetDtype: 'F16',
      storageEncoding: 'offset_binary',
      scaleSemantics: 'step',
      storageShape: [2, 2],
      quantAxis: 0,
      storageBlockSize: 4,
      storageLaneOrder: [0, 1, 1, 3],
      scaleSource: { shard: 1, offset: 0, size: 8 },
    },
  }, 'matrix/invalid-lane-order'),
  /invalid LiteRT storageLaneOrder/
);

console.log('source-transform-contract-matrix.test: ok');
