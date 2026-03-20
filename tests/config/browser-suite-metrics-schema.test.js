import assert from 'node:assert/strict';

import {
  BROWSER_SUITE_METRICS_SCHEMA_VERSION,
  DEFAULT_BROWSER_SUITE_METRICS,
  validateBrowserSuiteMetrics,
} from '../../src/config/schema/browser-suite-metrics.schema.js';

const metrics = validateBrowserSuiteMetrics({
  ...DEFAULT_BROWSER_SUITE_METRICS,
  schemaVersion: BROWSER_SUITE_METRICS_SCHEMA_VERSION,
  suite: 'inference',
  prompt: 'Hello world.',
  executionContractArtifact: {
    ok: true,
    checks: [],
    errors: [],
  },
  layerPatternContractArtifact: {
    ok: true,
    checks: [],
    errors: [],
  },
  requiredInferenceFieldsArtifact: {
    ok: true,
    checks: [],
    errors: [],
  },
});

assert.equal(metrics.schemaVersion, 1);
assert.equal(metrics.source, 'doppler');
assert.equal(metrics.suite, 'inference');

assert.throws(
  () => validateBrowserSuiteMetrics({
    schemaVersion: 2,
    source: 'doppler',
    suite: 'inference',
  }),
  /schemaVersion must be 1/
);

console.log('browser-suite-metrics-schema.test: ok');
