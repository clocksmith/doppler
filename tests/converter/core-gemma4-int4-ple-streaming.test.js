// Streaming-path test for Gemma 4 INT4 PLE per-row quantization. Synthetic
// tensor > MAX_TENSOR_TYPED_ARRAY_BYTES triggers the largeTensorTransformer
// route; the converter must accumulate per-chunk companionData (F32 scales)
// into a single scaleSource blob and attach sourceTransform to the tensor
// location in the manifest.

import assert from 'node:assert/strict';

import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const { convertModel } = await import('../../src/converter/core.js');
const { createConverterConfig } = await import('../../src/config/schema/converter.schema.js');
const { DEFAULT_MANIFEST_INFERENCE } = await import('../../src/config/schema/index.js');

// Synthetic large PLE tensor: claim 4.37 GB F16 (matches gemma-4-E2B-it) so
// that it takes the streaming path.
const PLE_ROWS = 262144;
const PLE_COLS = 8960;
const SOURCE_BYTES = PLE_ROWS * PLE_COLS * 2; // F16 bytes

const model = {
  name: 'gemma4-ple-streaming-test',
  modelId: 'gemma4-ple-streaming-test',
  quantization: 'F16',
  tensors: [
    {
      name: 'model.language_model.embed_tokens_per_layer.weight',
      shape: [PLE_ROWS, PLE_COLS],
      dtype: 'F16',
      size: SOURCE_BYTES,
      offset: 0,
      sourcePath: '/tmp/fake-gemma4-ple.safetensors',
    },
  ],
  config: { model_type: 'gemma4_text' },
};

const baseOptions = {
  modelId: 'gemma4-ple-streaming-test',
  modelType: 'transformer',
  quantization: 'F16',
  quantizationInfo: {
    weights: 'f16', embeddings: 'f16', compute: 'f16', variantTag: 'f16',
  },
  architecture: {
    numLayers: 35, hiddenSize: 2, intermediateSize: 8, numAttentionHeads: 1,
    numKeyValueHeads: 1, headDim: 2, vocabSize: PLE_ROWS, maxSeqLen: 8, ropeTheta: 1e6,
  },
  inference: { ...DEFAULT_MANIFEST_INFERENCE },
  eosTokenId: 1,
  converterConfig: createConverterConfig(),
};

const writtenShards = [];
let capturedManifest = null;
const io = {
  async readTensorData() {
    throw new Error('readTensorData should not be called for oversized streamed tensors');
  },
  async writeShard(index, data) {
    writtenShards.push({ index, byteLength: data.byteLength });
    return `hash-${index}`;
  },
  async writeManifest(manifest) { capturedManifest = manifest; },
};

// Large PLE tensor streaming. Emit 4 chunks:
//   chunk 0: rows 0..65535         → INT4 bytes + 65536 F32 scales
//   chunk 1: rows 65536..131071    → INT4 bytes + 65536 F32 scales
//   chunk 2: rows 131072..196607   → INT4 bytes + 65536 F32 scales
//   chunk 3: rows 196608..262143   → INT4 bytes + 65536 F32 scales
// Total: PLE_ROWS rows of INT4 (PLE_COLS/2 bytes per row) + PLE_ROWS F32 scales.
const ROWS_PER_CHUNK = 65536;
const INT4_ROW_BYTES = PLE_COLS / 2;
const SCALE_ROW_BYTES = 4;
const EXPECTED_INT4_BYTES = PLE_ROWS * INT4_ROW_BYTES;
const EXPECTED_SCALE_BYTES = PLE_ROWS * SCALE_ROW_BYTES;

const shardSize = EXPECTED_INT4_BYTES + EXPECTED_SCALE_BYTES + 4096; // fit everything in one shard

const sharedSourceTransform = {
  kind: 'litert_axis_dequant',
  scheme: 'per_axis_affine',
  sourceDtype: 'INT4',
  targetDtype: 'F16',
  storageEncoding: 'offset_binary',
  scaleSemantics: 'step',
  storageShape: [PLE_ROWS, PLE_COLS],
  quantAxis: 1,
};

const result = await convertModel(model, io, {
  ...baseOptions,
  shardSize,
  async largeTensorTransformer(input) {
    assert.equal(input.tensor.name, 'model.language_model.embed_tokens_per_layer.weight');
    for (let chunkStart = 0; chunkStart < PLE_ROWS; chunkStart += ROWS_PER_CHUNK) {
      const rowCount = Math.min(ROWS_PER_CHUNK, PLE_ROWS - chunkStart);
      // Fill INT4 chunk with predictable pattern (byte value = chunk index + row offset)
      const int4Bytes = new Uint8Array(rowCount * INT4_ROW_BYTES);
      for (let i = 0; i < int4Bytes.length; i++) {
        int4Bytes[i] = (chunkStart + (i & 0xff)) & 0xff;
      }
      // F32 scales: one per row
      const scales = new Float32Array(rowCount);
      for (let r = 0; r < rowCount; r++) {
        scales[r] = (chunkStart + r + 1) * 1e-5;
      }
      const scaleBytes = new Uint8Array(scales.buffer, scales.byteOffset, scales.byteLength);
      await input.writeChunk({
        tensorData: int4Bytes,
        companionData: scaleBytes,
        outDtype: 'F16',
        outLayout: 'row',
        sourceTransform: sharedSourceTransform,
      });
    }
    return { outDtype: 'F16', outLayout: 'row' };
  },
});

// --- Assertions ---
assert.ok(capturedManifest, 'manifest must be written');
const entry = capturedManifest.tensors['model.language_model.embed_tokens_per_layer.weight'];
assert.ok(entry, 'PLE tensor must appear in manifest');
assert.equal(entry.dtype, 'F16', 'dtype must be F16 (logical post-dequant)');
assert.equal(entry.layout, 'row', 'layout must be row');
assert.deepEqual(entry.shape, [PLE_ROWS, PLE_COLS]);
assert.equal(entry.size, EXPECTED_INT4_BYTES, 'PLE body size must match INT4 bytes');

const st = entry.sourceTransform;
assert.ok(st, 'PLE entry must carry sourceTransform');
assert.equal(st.kind, 'litert_axis_dequant');
assert.equal(st.sourceDtype, 'INT4');
assert.equal(st.scaleSemantics, 'step');
assert.equal(st.quantAxis, 1);
assert.deepEqual(st.storageShape, [PLE_ROWS, PLE_COLS]);

assert.ok(st.scaleSource, 'scaleSource must be populated after streaming');
assert.equal(typeof st.scaleSource.shard, 'number');
assert.equal(typeof st.scaleSource.offset, 'number');
assert.equal(st.scaleSource.size, EXPECTED_SCALE_BYTES, 'scaleSource.size must equal rows * 4 bytes');

// total bytes written = INT4 body + scales companion
const totalWritten = writtenShards.reduce((sum, s) => sum + s.byteLength, 0);
assert.equal(totalWritten, EXPECTED_INT4_BYTES + EXPECTED_SCALE_BYTES, 'shard byte sum must equal body + companion');

console.log('core-gemma4-int4-ple-streaming.test: ok');
