import assert from 'node:assert/strict';
import {
  DEFAULT_DEFERRED_ROUNDING_WINDOW_TOKENS,
  DEFAULT_FINITENESS_ABS_THRESHOLD,
  resolveDeferredRoundingWindowTokens,
  resolveRangeAwareSelectiveWideningConfig,
  shouldRunFinitenessGuard,
} from '../../src/inference/pipelines/text/finiteness-policy.js';

{
  const policy = resolveRangeAwareSelectiveWideningConfig(undefined);
  assert.equal(policy.enabled, true);
  assert.equal(policy.includeNonFinite, true);
  assert.equal(policy.absThreshold, DEFAULT_FINITENESS_ABS_THRESHOLD);
}

{
  const policy = resolveRangeAwareSelectiveWideningConfig({
    rangeAwareSelectiveWidening: {
      enabled: false,
      includeNonFinite: false,
      absThreshold: 64000,
    },
  });
  assert.equal(policy.enabled, false);
  assert.equal(policy.includeNonFinite, false);
  assert.equal(policy.absThreshold, 64000);
}

{
  const policy = resolveRangeAwareSelectiveWideningConfig({
    rangeAwareSelectiveWidening: {
      absThreshold: -1,
    },
  });
  assert.equal(policy.absThreshold, DEFAULT_FINITENESS_ABS_THRESHOLD);
}

assert.equal(resolveDeferredRoundingWindowTokens(undefined), DEFAULT_DEFERRED_ROUNDING_WINDOW_TOKENS);
assert.equal(resolveDeferredRoundingWindowTokens({ deferredRoundingWindowTokens: 4.9 }), 4);
assert.equal(resolveDeferredRoundingWindowTokens({ deferredRoundingWindowTokens: 0 }), DEFAULT_DEFERRED_ROUNDING_WINDOW_TOKENS);
assert.equal(shouldRunFinitenessGuard('f16', undefined), true);
assert.equal(shouldRunFinitenessGuard('f32', undefined), false);
assert.equal(shouldRunFinitenessGuard('f16', { rangeAwareSelectiveWidening: { enabled: false } }), false);

console.log('finiteness-policy.test: ok');
