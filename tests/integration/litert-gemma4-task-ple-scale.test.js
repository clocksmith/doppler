// Integration test for Gemma 4 E2B LiteRT `.task` direct-source PLE loading.
//
// Verifies that Doppler can load `gemma-4-E2B-it-web.task` (the LiteRT-LM
// Gemma 4 E2B artifact) via `loadMode=memory` on the Node surface, including
// the per-layer-embedding (PLE) tensors that were previously blocked.
//
// Empirical facts, taken from a FlatBuffer walk of the actual
// `gemma-4-E2B-it-web.task` file on disk (2026-04-16, size 2,003,697,664
// bytes, root `TFL3`):
//
//   1. The file contains exactly one subgraph, `GEMMA4_2P3B`, with 1745
//      tensors and zero operators. It is a pure weight bag packaged as a
//      TFLite FlatBuffer, not an executable TFLite graph.
//   2. 70 tensors match the `per_layer_embeddings` prefix (35 layers x 2
//      tensors per layer: `.w` and `.w_quantized_scale`).
//   3. `transformer.layer_N.per_layer_embeddings.w` is UINT8 with shape
//      `[33554432]` (a flattened 1-D storage shape). Logical shape is
//      `[vocabSizePerLayerInput=262144, hiddenSizePerLayerInput=256]` at
//      INT4 (0.5 bytes/element = 33,554,432 bytes).
//   4. `transformer.layer_N.per_layer_embeddings.w_quantized_scale` is UINT8
//      with shape `[1048576]`. 1,048,576 bytes = 262,144 x 4, holding one
//      F32 row-scale per embedding row packed in a UINT8 byte container.
//   5. Both tensors have `Tensor.quantization` entirely absent from the
//      FlatBuffer vtable — not "scale.length !== 1", not
//      "quantized_dimension !== 0" — *absent*. There is no per-channel
//      metadata to relax.
//
// The dequant convention comes from the MediaPipe symmetric quantization
// path (quantization_util.py `reduce_precision`): signed INT4 values in
// [-8, 7], scale = max(|row|) / 7, dequant = int4_value * scale,
// zero_point = 0. The UINT8 FlatBuffer dtype is a byte container — the
// scale companion bytes are native F32 (no affine dequant needed).
//
// The test is skipped when the local `.task` artifact is not present so CI
// environments without the ~1.9 GB source file stay green.

import assert from 'node:assert/strict';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, '..', '..');
const TASK_PATH = path.join(
  repoRoot,
  'models',
  'source',
  'litert',
  'gemma-4-e2b-it',
  'gemma-4-E2B-it-web.task'
);

async function pathExists(target) {
  try {
    await fs.access(target);
    return true;
  } catch {
    return false;
  }
}

const available = await pathExists(TASK_PATH);
if (!available) {
  console.log(
    'litert-gemma4-task-ple-scale.test: skipped (source artifact not available at '
    + TASK_PATH + ')'
  );
  process.exit(0);
}

const { resolveNodeSourceRuntimeBundle } = await import(
  '../../src/tooling/node-source-runtime.js'
);

const bundle = await resolveNodeSourceRuntimeBundle({
  inputPath: TASK_PATH,
  modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
  verifyHashes: false,
  runtimeConfig: null,
});

assert.ok(bundle, 'resolveNodeSourceRuntimeBundle must return a bundle.');
assert.ok(bundle.manifest, 'Bundle must have a manifest.');
assert.strictEqual(
  bundle.manifest.tensorCount,
  576,
  `Expected 576 tensors, got ${bundle.manifest.tensorCount}.`
);
assert.strictEqual(bundle.manifest.tokenizer?.type, 'sentencepiece');
assert.strictEqual(bundle.manifest.tokenizer?.sentencepieceModel, 'TOKENIZER_MODEL');
assert.strictEqual(bundle.manifest.tokenizer?.padTokenId, 0);
assert.strictEqual(bundle.manifest.tokenizer?.eosTokenId, 1);
assert.deepStrictEqual(bundle.manifest.tokenizer?.eosTokens, [1, 106, 50]);
assert.strictEqual(bundle.manifest.tokenizer?.bosTokenId, 2);
assert.strictEqual(bundle.manifest.tokenizer?.unkTokenId, 3);
assert.strictEqual(bundle.manifest.tokenizer?.addBosToken, false);
assert.strictEqual(bundle.manifest.tokenizer?.addEosToken, false);

const tensors = bundle.manifest.tensors;
assert.ok(tensors, 'Manifest must have a tensors map.');

const pleKey = 'model.language_model.layers.0.embed_tokens_per_layer.weight';
const pleTensor = tensors[pleKey];
assert.ok(pleTensor, `PLE tensor "${pleKey}" must exist in the manifest.`);
assert.deepStrictEqual(
  pleTensor.shape,
  [262144, 256],
  `PLE tensor shape must be [262144, 256], got ${JSON.stringify(pleTensor.shape)}.`
);
assert.strictEqual(pleTensor.dtype, 'F16', `PLE tensor dtype must be F16.`);
assert.strictEqual(
  pleTensor.sourceTransform?.kind,
  'litert_axis_dequant',
  'PLE sourceTransform.kind must be litert_axis_dequant.'
);
assert.strictEqual(
  pleTensor.sourceTransform?.sourceDtype,
  'INT4',
  'PLE sourceTransform.sourceDtype must be INT4.'
);
assert.strictEqual(
  pleTensor.sourceTransform?.scaleSemantics,
  'step',
  'PLE sourceTransform.scaleSemantics must be step.'
);
assert.strictEqual(
  pleTensor.sourceTransform?.quantAxis,
  1,
  'PLE sourceTransform.quantAxis must be 1.'
);
assert.strictEqual(
  pleTensor.sourceTransform?.scaleCompanionDtype,
  undefined,
  'PLE sourceTransform must NOT have scaleCompanionDtype (scale bytes are native F32).'
);
assert.strictEqual(
  pleTensor.sourceTransform?.scaleCompanionDequant,
  undefined,
  'PLE sourceTransform must NOT have scaleCompanionDequant.'
);

console.log('litert-gemma4-task-ple-scale.test: ok (PLE gap closed, 576 tensors loaded)');
