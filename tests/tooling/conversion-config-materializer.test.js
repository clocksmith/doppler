// The materializer refreshes a manifest's inference block from a conversion
// config. Its contract: config-owned fields (execution graph, session decode
// loop, compute/kv defaults, inference sections) flow through unchanged, model
// identity comes from the manifest, and the result is a resolvable execution-v1
// graph. Assertions compare against the config the output is derived from, not
// against copied tuning literals.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

import { expandExecutionV1 } from '../../src/config/schema/execution-v1.schema.js';
import {
  inferConversionConfigModelId,
  resolveMaterializedManifestFromConversionConfig,
} from '../../src/tooling/conversion-config-materializer.js';

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

// inferConversionConfigModelId: explicit modelBaseId wins, file stem otherwise.
assert.equal(
  inferConversionConfigModelId('src/config/conversion/x/file-stem.json', { output: { modelBaseId: '  declared-id ' } }),
  'declared-id'
);
assert.equal(inferConversionConfigModelId('src/config/conversion/x/file-stem.json', { output: {} }), 'file-stem');
assert.equal(inferConversionConfigModelId('src/config/conversion/x/file-stem.json', { output: { modelBaseId: '   ' } }), 'file-stem');

const FIXTURES = [
  'gemma3/gemma-3-270m-it-q4k-ehf16-af32',
  'gemma3/gemma-3-1b-it-q4k-ehf16-af32',
  'gemma4/gemma-4-e2b-it-q4k-ehf16-af32',
  'qwen3/qwen-3-5-0-8b-q4k-ehaf16',
  'qwen3/qwen-3-5-2b-q4k-ehaf16',
  'embeddinggemma/google-embeddinggemma-300m-q4k-ehf16-af32',
];

function assertMaterializedFromConfig(conversionConfig, manifest, materialized, label) {
  assert.equal(materialized.modelId, manifest.modelId, `${label}: modelId comes from the manifest`);
  assert.equal(materialized.modelType, manifest.modelType, `${label}: modelType comes from the manifest`);
  assert.equal(materialized.architecture, manifest.architecture, `${label}: architecture is passed through`);

  const inference = materialized.inference;
  assert.equal(inference?.schema, 'doppler.execution/v1', `${label}: materialized inference is execution-v1`);
  assert.equal(inference.defaultKernelPath, undefined, `${label}: execution-v1 output omits defaultKernelPath`);

  const execution = inference.execution;
  for (const field of ['inlineKernelPath', 'kernels', 'preLayer', 'decode', 'prefill', 'postLayer']) {
    assert.deepEqual(execution?.[field], conversionConfig.execution[field], `${label}: execution.${field} is config-owned`);
  }
  assert.ok(expandExecutionV1(execution).length > 0, `${label}: materialized graph must expand`);

  for (const field of ['decodeLoop', 'kvcache']) {
    if (conversionConfig.session?.[field] !== undefined) {
      assert.deepEqual(inference.session?.[field], conversionConfig.session[field], `${label}: session.${field} is config-owned`);
    }
  }
  assert.deepEqual(inference.session?.compute?.defaults, conversionConfig.session?.compute?.defaults, `${label}: compute defaults`);

  for (const [section, value] of Object.entries(conversionConfig.inference ?? {})) {
    if (!value || typeof value !== 'object' || Array.isArray(value)) continue;
    for (const [field, fieldValue] of Object.entries(value)) {
      assert.deepEqual(inference[section]?.[field], fieldValue, `${label}: inference.${section}.${field} is config-owned`);
    }
  }
}

let exercised = 0;
for (const fixture of FIXTURES) {
  const modelId = path.basename(fixture);
  const manifestPath = path.join('models/local', modelId, 'manifest.json');
  if (!fs.existsSync(manifestPath)) {
    console.log(`conversion-config-materializer.test: skipped ${modelId} (local model missing)`);
    continue;
  }
  const configPath = `src/config/conversion/${fixture}.json`;
  const conversionConfig = readJson(configPath);
  const manifest = readJson(manifestPath);
  assert.equal(inferConversionConfigModelId(configPath, conversionConfig), manifest.modelId, `${modelId}: config id matches manifest`);
  const materialized = resolveMaterializedManifestFromConversionConfig(conversionConfig, manifest);
  assertMaterializedFromConfig(conversionConfig, manifest, materialized, modelId);
  exercised += 1;

  // Changing the config changes the output: the config, not a runtime default
  // or the stale manifest, is the source of these values.
  const edited = structuredClone(conversionConfig);
  edited.session.decodeLoop.batchSize += 3;
  const firstKernelKey = Object.keys(edited.execution.kernels)[0];
  edited.execution.kernels[firstKernelKey].constants = { MATERIALIZER_PROBE: 1 };
  const rematerialized = resolveMaterializedManifestFromConversionConfig(edited, manifest);
  assert.equal(rematerialized.inference.session.decodeLoop.batchSize, conversionConfig.session.decodeLoop.batchSize + 3);
  assert.deepEqual(rematerialized.inference.execution.kernels[firstKernelKey].constants, { MATERIALIZER_PROBE: 1 });
}

// Synthetic manifest: the contract holds without local artifacts, using a
// checked-in config and the minimal manifest fields the materializer reads.
{
  const configPath = 'src/config/conversion/gemma3/gemma-3-270m-it-q4k-ehf16-af32.json';
  const conversionConfig = readJson(configPath);
  const manifest = {
    modelId: 'synthetic-materializer-fixture',
    modelType: 'transformer',
    quantizationInfo: { weights: 'q4k' },
    architecture: { numLayers: 18, headDim: 256 },
    config: { model_type: 'gemma3_text' },
    tensors: {},
  };
  const materialized = resolveMaterializedManifestFromConversionConfig(conversionConfig, manifest);
  assertMaterializedFromConfig(conversionConfig, manifest, materialized, 'synthetic');
}

console.log(`conversion-config-materializer.test: ok (${exercised} local fixtures)`);
