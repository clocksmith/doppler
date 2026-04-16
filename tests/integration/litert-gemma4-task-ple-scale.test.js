// Diagnostic regression test for the Gemma 4 E2B LiteRT `.task` direct-source
// PLE scale-companion gap.
//
// When Doppler tries to load `gemma-4-E2B-it-web.task` (the LiteRT-LM-packaged
// Gemma 4 E2B artifact) via `loadMode=memory` on the Node surface, the parser
// reaches `createLiteRTAxisTensor` for every `transformer.layer_N.per_layer_embeddings.w`
// tensor and throws.
//
// Empirical facts, taken from a FlatBuffer walk of the actual
// `gemma-4-E2B-it-web.task` file on disk (2026-04-16, size 2,003,697,664 bytes,
// root `TFL3`):
//
//   1. The file contains exactly one subgraph, `GEMMA4_2P3B`, with 1745
//      tensors and **zero operators**. It is a pure weight bag packaged as a
//      TFLite FlatBuffer, not an executable TFLite graph.
//   2. 70 tensors match the `per_layer_embeddings` prefix (35 layers × 2
//      tensors per layer: `.w` and `.w_quantized_scale`).
//   3. `transformer.layer_N.per_layer_embeddings.w` is UINT8 with shape
//      `[33554432]` (a flattened 1-D storage shape — logically
//      `[vocabSizePerLayerInput=262144, hiddenSizePerLayerInput=128]`), backed
//      by an offload buffer of 33,554,432 bytes.
//   4. `transformer.layer_N.per_layer_embeddings.w_quantized_scale` is UINT8
//      with shape `[1048576]`. 1,048,576 bytes = 262,144 × 4, exactly the
//      right size to hold one F32 row-scale per embedding row stored inside
//      a UINT8 byte container.
//   5. **Both tensors have `Tensor.quantization` entirely absent from the
//      FlatBuffer vtable** (not "scale.length !== 1", not
//      "quantized_dimension !== 0" — *absent*). There is nothing for
//      `readTensorQuantization` in `src/formats/tflite/types.js` to relax.
//   6. The model-level `metadata` table has 4 entries:
//        `odml.infra.proto.LlmParameters` (buffer=1, size=209 bytes),
//        `spm_vocab_model` (buffer=2), `odml.infra.LlmModelType` (buffer=52),
//        `backend` (buffer=3). The `LlmParameters` proto carries model
//        architecture fields (`hidden_size=1536`, `num_layers=35`,
//        `per_layer_hidden_size=128`, `vocab_size=262144`, EOS/turn tokens)
//        — **not** per-tensor quantization factors.
//
// This refutes the earlier "relax the per-channel assertion in
// `readTensorQuantization`" hypothesis. There is no canonical per-channel
// quantization metadata to relax — the FlatBuffer simply does not carry it
// for these tensors. A parser fix at `src/formats/tflite/types.js:379` would
// have nothing to read.
//
// It also contradicts the earlier claim that "PLE in Gemma 3n/Gemma 4 is a
// separate compiled TFLite subgraph (TF_LITE_PER_LAYER_EMBEDDER)". For this
// particular `.task` file there is no separate embedder subgraph at all —
// `GEMMA4_2P3B` is a 0-operator weight container. The convention for
// dequantizing `per_layer_embeddings.w` using the bytes in
// `per_layer_embeddings.w_quantized_scale` therefore lives in the LiteRT-LM
// runtime source tree, not inside the `.task` artifact. An earlier
// Doppler-side experiment that reinterpreted the companion bytes as packed
// F32 row-scales with `zero_point=128` loaded and ran end-to-end but
// produced garbled output ("蔗izmiFCO🥥" for "Hello"), so the convention
// must be something more than a naive reinterpret.
//
// Paths that can actually close this lane (in decreasing order of cost):
//
//   a. Check out `github.com/google-ai-edge/LiteRT-LM` and search its C++
//      text runtime for `per_layer_embeddings` / `_quantized_scale` to
//      recover the exact dequant convention, then port it into
//      `src/tooling/litert-package-runtime.js:createLiteRTAxisTensor`.
//   b. Run a reference LiteRT-LM text runner locally against the same
//      `.task` file and dump F32 intermediates at the embedder output for
//      one or two known tokens, then fit the dequant convention against
//      those ground-truth values.
//   c. A published LiteRT-LM spec document naming the packing convention.
//
// Until one of those lands, this test pins the current failure shape so:
//
//   a. A future fix can flip the test from "expect-throw" to "expect-success"
//      in a single commit, and
//   b. A future refactor of the LiteRT package runtime will surface this gap
//      immediately if it accidentally changes the error contract.
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

let caught = null;
try {
  await resolveNodeSourceRuntimeBundle({
    inputPath: TASK_PATH,
    modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
    verifyHashes: false,
    runtimeConfig: null,
  });
} catch (error) {
  caught = error;
}

assert.ok(
  caught instanceof Error,
  'Loading gemma-4-E2B-it-web.task must currently fail at the PLE scale-companion check. '
  + 'If this assertion starts failing, the LiteRT .task PLE gap has been closed — '
  + 'update this test to assert success and remove the diagnostic comment in '
  + 'src/tooling/litert-package-runtime.js:createLiteRTAxisTensor.'
);

assert.match(
  caught.message,
  /per_layer_embeddings\.w_quantized_scale/,
  'Expected failure to name the PLE scale-companion tensor. Actual: ' + caught.message
);

assert.match(
  caught.message,
  /affine_dequant metadata|scale-of-scale convention/,
  'Expected failure to reference the missing affine_dequant metadata / scale-of-scale convention. Actual: '
  + caught.message
);

console.log('litert-gemma4-task-ple-scale.test: ok (gap pinned)');
