import assert from 'node:assert/strict';

import { createDopplerConfig } from '../../src/config/schema/index.js';
import { resolvePrefillChunkSubmitMode } from '../../src/inference/pipelines/text/generator.js';

function createRuntimeConfig(prefillChunkSubmitMode) {
  const runtimeConfig = createDopplerConfig().runtime;
  runtimeConfig.inference.session.prefillChunkSubmitMode = prefillChunkSubmitMode;
  return runtimeConfig;
}

{
  const mode = resolvePrefillChunkSubmitMode(
    createRuntimeConfig('async'),
    { sessionSettings: { prefillChunkSubmitMode: 'sync' } }
  );
  assert.equal(mode, 'async', 'runtime session value must override manifest session value.');
}

{
  const mode = resolvePrefillChunkSubmitMode(
    { inference: { session: { prefillChunkSubmitMode: null } } },
    { sessionSettings: { prefillChunkSubmitMode: 'sync' } }
  );
  assert.equal(mode, 'sync', 'manifest session value must be used only when runtime leaves the field unset.');
}

{
  assert.throws(
    () => resolvePrefillChunkSubmitMode(
      { inference: { session: { prefillChunkSubmitMode: null } } },
      { sessionSettings: { prefillChunkSubmitMode: null } }
    ),
    /runtime\.inference\.session\.prefillChunkSubmitMode is required/
  );
}

{
  assert.throws(
    () => resolvePrefillChunkSubmitMode(
      { inference: { session: { prefillChunkSubmitMode: 'later' } } },
      { sessionSettings: { prefillChunkSubmitMode: 'sync' } }
    ),
    /prefillChunkSubmitMode must be "sync" or "async"/
  );
}

console.log('generator-prefill-submit-mode.test: ok');
