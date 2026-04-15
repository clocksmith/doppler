// Diagnostic regression test for the Gemma 4 E2B LiteRT `.task` direct-source
// PLE scale-companion gap.
//
// When Doppler tries to load `gemma-4-E2B-it-web.task` (the LiteRT-LM-packaged
// Gemma 4 E2B artifact) via `loadMode=memory` on the Node surface, the parser
// reaches `createLiteRTAxisTensor` for every `transformer.layer_N.per_layer_embeddings.w`
// tensor and throws because:
//
//   1. `per_layer_embeddings.w` is UINT8
//   2. `per_layer_embeddings.w_quantized_scale` is ALSO UINT8 (packed scales)
//   3. The TFLite FlatBuffer carries no quantization table for the `_quantized_scale`
//      tensor, so `readTensorQuantization` in `src/formats/tflite/types.js` never
//      produces a sourceTransform for it
//   4. There are no sibling F32/F16 min/max/range metadata tensors
//   5. The FlatBuffer-level metadata table is empty
//
// So the LiteRT-LM convention for deriving the UINT8-scale-companion-of-UINT8-weight
// dequant (the "scale of the scale") must come from somewhere we are not yet
// reading. This test pins the current failure shape so:
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
