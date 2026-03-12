import assert from 'node:assert/strict';

import { shouldRetryWithFinitenessFallback } from '../../src/inference/pipelines/text/generator.js';

assert.equal(
  shouldRetryWithFinitenessFallback({ name: 'FinitenessError', message: 'bounds exceeded' }),
  true
);

assert.equal(
  shouldRetryWithFinitenessFallback(new Error(
    '[Sampling] Logits has no finite candidate logits after masking the pad token. Upstream decode likely produced NaN/Inf or an all-masked distribution.'
  )),
  true
);

assert.equal(
  shouldRetryWithFinitenessFallback(new Error(
    '[Sampling] Softmax produced no finite candidate probabilities. Upstream decode likely produced NaN/Inf logits.'
  )),
  true
);

assert.equal(
  shouldRetryWithFinitenessFallback(new Error('ordinary failure')),
  false
);

// Edge: null/undefined error input
assert.equal(shouldRetryWithFinitenessFallback(null), false);
assert.equal(shouldRetryWithFinitenessFallback(undefined), false);

// Edge: raw string input (exercises typeof error === 'string' branch)
assert.equal(
  shouldRetryWithFinitenessFallback(
    '[Sampling] Logits has no finite candidate logits after masking the pad token.'
  ),
  true
);
assert.equal(shouldRetryWithFinitenessFallback('random string'), false);

// Edge: [Sampling] prefix but unrecognized message
assert.equal(
  shouldRetryWithFinitenessFallback(new Error('[Sampling] Unknown sampling issue')),
  false
);

console.log('generator-finiteness-fallback.test: ok');
