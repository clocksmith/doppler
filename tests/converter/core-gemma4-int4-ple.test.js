// Contract + round-trip test for Gemma 4 per-layer-embeddings INT4 per-row
// symmetric quantization path in `transformTensorBytes`. Convention matches
// the MediaPipe LiteRT-LM composites extracted from gemma-4-E2B-it.litertlm
// (quantizedDimension=0, one F32 scale per vocab row, zero_point=0).

import assert from 'node:assert/strict';
import {
  transformTensorBytes,
  isGemma4PerLayerEmbedTensor,
} from '../../src/converter/core.js';
import {
  dequantizeInt4PerRowSymmetric,
  float32ToFloat16,
} from '../../src/converter/quantizer.js';

// --- 1. isGemma4PerLayerEmbedTensor routing ---
assert.equal(isGemma4PerLayerEmbedTensor('model.language_model.embed_tokens_per_layer.weight'), true);
assert.equal(isGemma4PerLayerEmbedTensor('language_model.layers.0.per_layer_embeddings.weight'), true);
assert.equal(isGemma4PerLayerEmbedTensor('model.language_model.embed_tokens.weight'), false);
assert.equal(isGemma4PerLayerEmbedTensor('self_attn.k_proj.weight'), false);
assert.equal(isGemma4PerLayerEmbedTensor(null), false);

// --- 2. transformTensorBytes returns correct envelope for PLE tensor ---
// Build a small synthetic PLE tensor with HF layout (rows=vocab >> cols=embed).
// The shape guard rejects tensors where rows <= cols to avoid mis-quantizing
// GGUF-transposed [embed, vocab] layouts on the wrong axis.
const rows = 8;
const cols = 4;
const f32 = new Float32Array([
  // row 0: uniform positive sweep, 4 cols
  0.0,    0.10,   0.20,   0.30,
  // row 1: uniform negative sweep
  -0.30,  -0.20,  -0.10,  0.0,
  // row 2: small magnitudes
  0.001,  0.002,  0.003,  0.004,
  // row 3: outlier-dominated (most magnitudes will clip to 0)
  1.0,    0.5,    0.0625, 0.0078125,
  // row 4: symmetric around 0, mid magnitude
  -0.5,   -0.1,   0.1,    0.5,
  // row 5: uniform mid-positive
  0.25,   0.50,   0.75,   1.00,
  // row 6: very small uniform
  1e-5,   2e-5,   3e-5,   4e-5,
  // row 7: alternating sign
  -0.3,   0.3,    -0.2,   0.2,
]);
const f16 = new Uint16Array(f32.length);
for (let i = 0; i < f32.length; i++) f16[i] = float32ToFloat16(f32[i]);
const srcBytes = new Uint8Array(f16.buffer, f16.byteOffset, f16.byteLength);

const tensor = {
  name: 'model.language_model.embed_tokens_per_layer.weight',
  shape: [rows, cols],
  dtype: 'F16',
};
const result = transformTensorBytes(tensor, srcBytes, { targetQuant: 'q4k' });

// Logical (post-dequant) dtype is F16; INT4 is the storage form carried in
// sourceTransform.sourceDtype. This matches the runtime's location contract
// where location.dtype describes the runtime-visible tensor after dequant.
assert.equal(result.outDtype, 'F16', 'outDtype must be F16 (logical, post-dequant)');
assert.equal(result.outLayout, 'row', 'outLayout must be row');
assert.equal(result.tensorTargetQuant, 'int4_per_row_ple');
assert.ok(result.tensorData instanceof Uint8Array, 'tensorData must be Uint8Array');
assert.equal(result.tensorData.byteLength, (rows * cols) / 2, 'INT4 storage = rows*cols/2 bytes');
assert.ok(result.companionData instanceof Uint8Array, 'companionData must be Uint8Array');
assert.equal(result.companionData.byteLength, rows * 4, 'scales companion = rows*4 bytes');

const st = result.sourceTransform;
assert.ok(st, 'sourceTransform must be emitted');
assert.equal(st.kind, 'litert_axis_dequant');
assert.equal(st.scheme, 'per_axis_affine');
assert.equal(st.sourceDtype, 'INT4');
assert.equal(st.targetDtype, 'F16');
assert.equal(st.storageEncoding, 'offset_binary');
assert.equal(st.scaleSemantics, 'step');
assert.deepEqual(st.storageShape, [rows, cols]);
assert.equal(st.quantAxis, 1, 'quantAxis=1 keeps storage matching logical shape');
assert.equal(st.scaleSource, undefined, 'scaleSource is filled by the caller, not the quantizer');

// --- 3. Round-trip: dequant of quantized bytes matches original within per-row quantization error ---
const scales = new Float32Array(
  result.companionData.buffer,
  result.companionData.byteOffset,
  result.companionData.byteLength / 4
);
const restored = dequantizeInt4PerRowSymmetric(result.tensorData, scales, [rows, cols]);

// Rows 0 and 1 span the full [-max, +max] range uniformly. Error budget is
// F16 source precision (~1e-3 for values near 0.35) plus at most one INT4
// quant step (scale = max/7). Source went F32→F16→F32→INT4, so compare
// against the F16-round-tripped reference, not the original F32.
const { float16ToFloat32 } = await import('../../src/converter/quantizer.js');
const srcRef = new Uint16Array(srcBytes.buffer, srcBytes.byteOffset, srcBytes.byteLength / 2);
const f16Ref = new Float32Array(srcRef.length);
for (let i = 0; i < srcRef.length; i++) f16Ref[i] = float16ToFloat32(srcRef[i]);

for (let c = 0; c < cols; c++) {
  const scale0 = scales[0];
  const scale1 = scales[1];
  const err0 = Math.abs(restored[0 * cols + c] - f16Ref[0 * cols + c]);
  const err1 = Math.abs(restored[1 * cols + c] - f16Ref[1 * cols + c]);
  assert.ok(err0 <= scale0 + 1e-6, `row 0 col ${c} err ${err0} exceeds one quant step ${scale0}`);
  assert.ok(err1 <= scale1 + 1e-6, `row 1 col ${c} err ${err1} exceeds one quant step ${scale1}`);
}

// Row 2 should have max abs error <= scale=0.008/7 (one quant step). No clipping.
const r2Err = Math.max(
  ...Array.from({ length: cols }, (_, c) => Math.abs(restored[2 * cols + c] - f16Ref[2 * cols + c]))
);
assert.ok(r2Err <= scales[2] + 1e-6, `row 2 max err ${r2Err} exceeds one quant step ${scales[2]}`);

// Row 3 is outlier-dominated (scale fits max=1.0); small values fall below the
// step and clip to 0. Outlier (col 0) must be preserved within one quant step.
assert.ok(
  Math.abs(restored[3 * cols + 0] - f16Ref[3 * cols + 0]) <= scales[3] + 1e-6,
  `row 3 outlier col 0 must be preserved within one quant step, got ${restored[3 * cols + 0]} vs ${f16Ref[3 * cols + 0]}`
);

// --- 4. Fallback: tensor with non-even cols falls back to q4k path ---
const oddTensor = {
  name: 'model.language_model.embed_tokens_per_layer.weight',
  shape: [rows, 7], // odd cols → cannot nibble-pack → fallback
  dtype: 'F32',
};
const oddBytes = new Uint8Array(new Float32Array(rows * 7).buffer);
const oddResult = transformTensorBytes(oddTensor, oddBytes, { targetQuant: 'q4k' });
assert.notEqual(oddResult.outDtype, 'INT4', 'odd-cols PLE tensor should NOT take INT4 path');

// --- 4b. Fallback: GGUF-layout PLE (rows <= cols) is rejected by the guard ---
// The INT4 per-row path requires rows > cols (HF convention [vocab, embed]).
// GGUF stores PLE transposed as [embed, vocab], which would quantize per the
// wrong axis. Guard must bail out.
const ggufLayoutTensor = {
  name: 'per_layer_token_embd.weight',
  shape: [4, 262144], // GGUF-like: embed << vocab
  dtype: 'F16',
};
const ggufBytes = new Uint8Array(new Uint16Array(4 * 262144).buffer);
const ggufResult = transformTensorBytes(ggufLayoutTensor, ggufBytes, { targetQuant: 'q4k' });
assert.notEqual(ggufResult.sourceDtype && ggufResult.tensorTargetQuant, 'int4_per_row_ple',
  'GGUF-layout PLE (rows<=cols) must NOT take INT4 per-row path');

// --- 5. Fallback: skipInt4PlePerRow option disables the branch ---
const skipResult = transformTensorBytes(tensor, srcBytes, {
  targetQuant: 'q4k',
  skipInt4PlePerRow: true,
});
assert.notEqual(skipResult.outDtype, 'INT4', 'skipInt4PlePerRow must disable the INT4 path');

console.log('core-gemma4-int4-ple.test: ok');
