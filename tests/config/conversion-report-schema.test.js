import assert from 'node:assert/strict';

import {
  CONVERSION_REPORT_SCHEMA_VERSION,
  DEFAULT_CONVERSION_REPORT,
  validateConversionReport,
} from '../../src/config/schema/conversion-report.schema.js';

{
  const report = validateConversionReport({
    ...DEFAULT_CONVERSION_REPORT,
    schemaVersion: CONVERSION_REPORT_SCHEMA_VERSION,
    modelId: 'gemma3-test',
    timestamp: '2026-03-06T12:00:00.000Z',
    result: {
      presetId: 'gemma3',
      modelType: 'transformer',
      outputDir: 'models/local/gemma3-test',
      shardCount: 1,
      tensorCount: 10,
      totalSize: 1024,
    },
    executionContractArtifact: {
      schemaVersion: 1,
      source: 'doppler',
      ok: true,
      checks: [],
      errors: [],
      session: null,
      steps: null,
    },
    executionV0GraphContractArtifact: {
      schemaVersion: 1,
      source: 'doppler',
      ok: true,
      checks: [],
      errors: [],
      stats: {
        prefillSteps: 0,
        decodeSteps: 0,
      },
    },
    layerPatternContractArtifact: {
      schemaVersion: 1,
      source: 'doppler',
      ok: true,
      checks: [],
      errors: [],
    },
    requiredInferenceFieldsArtifact: {
      schemaVersion: 1,
      source: 'doppler',
      scope: 'manifest',
      label: 'gemma3-test.inference',
      ok: true,
      checks: [],
      errors: [],
      stats: {
        fieldCases: 30,
        nullableCases: 10,
        nonNullableCases: 10,
      },
    },
  });

  assert.equal(report.schemaVersion, 1);
  assert.equal(report.suite, 'convert');
  assert.equal(report.command, 'convert');
}

{
  assert.throws(
    () => validateConversionReport({
      schemaVersion: 2,
      suite: 'convert',
      command: 'convert',
      modelId: 'bad',
      timestamp: '2026-03-06T12:00:00.000Z',
      source: 'doppler',
      result: {},
      manifest: null,
      executionContractArtifact: null,
      executionV0GraphContractArtifact: null,
      layerPatternContractArtifact: null,
      requiredInferenceFieldsArtifact: null,
    }),
    /schemaVersion must be 1/
  );
}

console.log('conversion-report-schema.test: ok');
