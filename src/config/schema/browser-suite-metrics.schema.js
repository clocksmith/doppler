function assertPlainObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`browser suite metrics: ${label} must be an object.`);
  }
}

function assertString(value, label) {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`browser suite metrics: ${label} must be a non-empty string.`);
  }
}

function assertNullablePlainObject(value, label) {
  if (value == null) return;
  assertPlainObject(value, label);
}

export const BROWSER_SUITE_METRICS_SCHEMA_VERSION = 1;

export const DEFAULT_BROWSER_SUITE_METRICS = Object.freeze({
  schemaVersion: BROWSER_SUITE_METRICS_SCHEMA_VERSION,
  source: 'doppler',
  suite: 'inference',
  executionContractArtifact: null,
  layerPatternContractArtifact: null,
  requiredInferenceFieldsArtifact: null,
});

export function validateBrowserSuiteMetrics(metrics) {
  assertPlainObject(metrics, 'metrics');
  if (metrics.schemaVersion !== BROWSER_SUITE_METRICS_SCHEMA_VERSION) {
    throw new Error(
      `browser suite metrics: schemaVersion must be ${BROWSER_SUITE_METRICS_SCHEMA_VERSION}.`
    );
  }
  if (metrics.source !== 'doppler') {
    throw new Error('browser suite metrics: source must be "doppler".');
  }
  assertString(metrics.suite, 'suite');
  assertNullablePlainObject(metrics.executionContractArtifact, 'executionContractArtifact');
  assertNullablePlainObject(metrics.layerPatternContractArtifact, 'layerPatternContractArtifact');
  assertNullablePlainObject(metrics.requiredInferenceFieldsArtifact, 'requiredInferenceFieldsArtifact');
  return metrics;
}
