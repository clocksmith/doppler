function assertPlainObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`conversion report: ${label} must be an object.`);
  }
}

function assertString(value, label) {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`conversion report: ${label} must be a non-empty string.`);
  }
}

function assertNullableString(value, label) {
  if (value === null || value === undefined) return;
  assertString(value, label);
}

function assertNullableFiniteNumber(value, label) {
  if (value === null || value === undefined) return;
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`conversion report: ${label} must be a finite number when provided.`);
  }
}

function assertNullablePlainObject(value, label) {
  if (value === null || value === undefined) return;
  assertPlainObject(value, label);
}

export const CONVERSION_REPORT_SCHEMA_VERSION = 1;

export const DEFAULT_CONVERSION_REPORT = Object.freeze({
  schemaVersion: CONVERSION_REPORT_SCHEMA_VERSION,
  suite: 'convert',
  command: 'convert',
  modelId: 'unknown',
  timestamp: '1970-01-01T00:00:00.000Z',
  source: 'doppler',
  result: {
    presetId: null,
    modelType: null,
    outputDir: null,
    shardCount: null,
    tensorCount: null,
    totalSize: null,
  },
  manifest: {
    quantization: null,
    quantizationInfo: null,
    inference: {
      presetId: null,
      schema: null,
      defaultKernelPath: null,
    },
  },
  executionContractArtifact: null,
});

export function validateConversionReport(report) {
  assertPlainObject(report, 'report');
  if (report.schemaVersion !== CONVERSION_REPORT_SCHEMA_VERSION) {
    throw new Error(
      `conversion report: schemaVersion must be ${CONVERSION_REPORT_SCHEMA_VERSION}.`
    );
  }
  if (report.suite !== 'convert') {
    throw new Error('conversion report: suite must be "convert".');
  }
  if (report.command !== 'convert') {
    throw new Error('conversion report: command must be "convert".');
  }
  if (report.source !== 'doppler') {
    throw new Error('conversion report: source must be "doppler".');
  }
  assertString(report.modelId, 'modelId');
  assertString(report.timestamp, 'timestamp');
  assertPlainObject(report.result, 'result');
  assertNullableString(report.result.presetId, 'result.presetId');
  assertNullableString(report.result.modelType, 'result.modelType');
  assertNullableString(report.result.outputDir, 'result.outputDir');
  assertNullableFiniteNumber(report.result.shardCount, 'result.shardCount');
  assertNullableFiniteNumber(report.result.tensorCount, 'result.tensorCount');
  assertNullableFiniteNumber(report.result.totalSize, 'result.totalSize');

  assertNullablePlainObject(report.manifest, 'manifest');
  if (report.manifest) {
    assertNullableString(report.manifest.quantization, 'manifest.quantization');
    assertNullablePlainObject(report.manifest.quantizationInfo, 'manifest.quantizationInfo');
    assertNullablePlainObject(report.manifest.inference, 'manifest.inference');
    if (report.manifest.inference) {
      assertNullableString(report.manifest.inference.presetId, 'manifest.inference.presetId');
      assertNullableString(report.manifest.inference.schema, 'manifest.inference.schema');
      assertNullableString(
        report.manifest.inference.defaultKernelPath,
        'manifest.inference.defaultKernelPath'
      );
    }
  }

  assertNullablePlainObject(report.executionContractArtifact, 'executionContractArtifact');
  return report;
}
