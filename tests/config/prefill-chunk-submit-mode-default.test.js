// Regression test for `runtime.inference.session.prefillChunkSubmitMode`.
//
// The prefill path chunks command-recorder submissions every N layers to
// bound intermediate buffer lifetime. By default it waits for each chunk to
// finish on GPU before the next is recorded (`sync`). Opting into `async`
// skips the CPU-GPU barrier when profile timings are not being collected —
// the GPU queue preserves ordering and deferred cleanup releases tracked
// buffers when work completes.
//
// This test locks:
//   1. The default mode is "sync" (preserves pre-opt-in behavior).
//   2. Profile overrides to "async" propagate through the runtime-config merge.
//   3. The value is a top-level session field alongside decodeLoop /
//      perLayerInputs, so config consumers can read it without touching
//      decodeLoop-specific state.

import assert from 'node:assert/strict';
import { createDopplerConfig } from '../../src/config/schema/doppler.schema.js';

{
  const defaults = createDopplerConfig();
  assert.strictEqual(
    defaults.runtime.inference.session.prefillChunkSubmitMode,
    'sync',
    'Default prefillChunkSubmitMode must be "sync" to preserve pre-opt-in behavior.'
  );
}

{
  const overridden = createDopplerConfig({
    runtime: {
      inference: {
        session: {
          prefillChunkSubmitMode: 'async',
        },
      },
    },
  });
  assert.strictEqual(
    overridden.runtime.inference.session.prefillChunkSubmitMode,
    'async',
    'Runtime override must propagate to the merged session config.'
  );
}

{
  // Override leaves adjacent session fields alone.
  const overridden = createDopplerConfig({
    runtime: {
      inference: {
        session: {
          prefillChunkSubmitMode: 'async',
        },
      },
    },
  });
  assert.ok(
    overridden.runtime.inference.session.kvcache !== undefined,
    'session.kvcache must remain populated when prefillChunkSubmitMode is overridden.'
  );
}

console.log('prefill-chunk-submit-mode-default.test: ok');
