import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

import { CONVERSION_REPORT_SCHEMA_VERSION } from '../../src/config/schema/conversion-report.schema.js';
import { createConverterConfig } from '../../src/config/schema/converter.schema.js';
import {
  createManifestConversionPostflightReceipt,
  validateManifestConversionPostflightReceipt,
} from '../../src/converter/manifest-conversion-postflight.js';
import { sha256Hex } from '../../src/utils/sha256.js';
import { stableSortObject } from '../../src/utils/stable-sort-object.js';

const stableDigest = (value) => `sha256:${sha256Hex(JSON.stringify(stableSortObject(value)))}`;
const shardHash = 'a'.repeat(64);
const conversionConfig = {
  modelType: 'transformer',
  quantization: { weights: 'f16' },
  sharding: { shardSizeBytes: 64 },
  manifest: { hashAlgorithm: 'sha256' },
  output: { textOnly: true },
};
const inference = { schema: 'doppler.execution/v1', execution: { schema: 'doppler.execution/v1' } };
const manifest = {
  modelId: 'postflight-test',
  modelType: 'transformer',
  quantization: 'F16',
  quantizationInfo: { weights: 'f16', embeddings: 'f16', compute: 'f32', variantTag: 'f16-af32' },
  hashAlgorithm: 'sha256',
  totalSize: 64,
  shards: [{ index: 0, filename: 'shard_00000.bin', size: 64, hash: shardHash, offset: 0 }],
  tensors: { 'model.embed_tokens.weight': { size: 64 } },
  tokenizer: { type: 'bundled', file: 'tokenizer.json' },
  metadata: { convertedAt: '2026-08-23T12:00:00.000Z' },
  inference,
};
const sourceCheckpointId = 'upstream/postflight-test';
const sourceFormat = 'safetensors';
const modalitySet = ['text'];
const materializationProfile = 'standard';
const resolvedConfig = createConverterConfig(conversionConfig);
const shardSetHash = stableDigest({ hashAlgorithm: 'sha256', shards: manifest.shards });
const weightPackHash = stableDigest({
  sourceCheckpointId,
  sourceFormat,
  modelType: manifest.modelType,
  modalitySet,
  quantizationInfo: manifest.quantizationInfo,
  materializationProfile,
  shardSetHash,
  sharding: { shardSizeBytes: 64 },
  output: { textOnly: true },
});
const weightPackId = `postflight-test-wp-${weightPackHash.slice(7, 19)}`;
const manifestVariantHash = stableDigest({
  weightPackId,
  modelType: manifest.modelType,
  inference,
  config: resolvedConfig.manifest,
});
manifest.artifactIdentity = {
  sourceCheckpointId,
  sourceRepo: 'upstream/postflight-test',
  sourceRevision: 'abc123',
  sourceFormat,
  conversionConfigDigest: stableDigest(conversionConfig),
  weightPackId,
  weightPackHash,
  shardSetHash,
  manifestVariantId: `postflight-test-mv-${manifestVariantHash.slice(7, 19)}`,
  modalitySet,
  materializationProfile,
  artifactCompleteness: 'complete',
};
const passingArtifact = { ok: true };
const conversionReport = {
  schemaVersion: CONVERSION_REPORT_SCHEMA_VERSION,
  suite: 'convert',
  command: 'convert',
  modelId: manifest.modelId,
  timestamp: manifest.metadata.convertedAt,
  startedAtUtc: '2026-08-23T11:59:00.000Z',
  completedAtUtc: '2026-08-23T12:00:00.000Z',
  durationMs: 60000,
  source: 'doppler',
  result: { modelType: 'transformer', outputDir: '/tmp/postflight', shardCount: 1, tensorCount: 1, totalSize: 64 },
  manifest: null,
  executionContractArtifact: passingArtifact,
  layerPatternContractArtifact: passingArtifact,
  requiredInferenceFieldsArtifact: passingArtifact,
};
const preflightReceipt = {
  schema: 'doppler.manifest-conversion-preflight-receipt/v1',
  modelId: manifest.modelId,
  entryPointId: 'text.generate',
  receiptDigest: stableDigest({ preflight: true }),
  semanticEvidence: { conversionConfigDigest: stableDigest(conversionConfig) },
  tensorClosureEvidence: { expectedTensorCount: 1 },
  sourceEvidence: {
    checkpointId: sourceCheckpointId,
    repository: 'upstream/postflight-test',
    revision: 'abc123',
  },
  conversionPlan: {
    inferenceDigest: stableDigest(inference),
    executionDigest: stableDigest(inference.execution),
  },
  dispositions: { headerPreflightPassed: true, conversionExecuted: false },
};
const inputs = {
  conversionConfig,
  conversionReport,
  conversionReportDigest: stableDigest(conversionReport),
  manifest,
  manifestDigest: stableDigest(manifest),
  preflightReceipt,
  shardObservations: [{ index: 0, filename: 'shard_00000.bin', size: 64, digest: `sha256:${shardHash}` }],
  artifactObservations: [{ role: 'tokenizer', path: 'tokenizer.json', size: 10, digest: stableDigest('tokenizer') }],
  policy: {
    schema: 'doppler.manifest-conversion-postflight/v1',
    author: { kind: 'tool', actor: 'postflight-test' },
  },
};

const receipt = createManifestConversionPostflightReceipt(inputs);
assert.equal(receipt.dispositions.conversionExecuted, true);
assert.equal(receipt.dispositions.physicalShardClosureVerified, true);
assert.equal(receipt.dispositions.qualificationStarted, false);
assert.equal(receipt.dispositions.packEligible, false);
assert.equal(receipt.conversionEvidence.durationMs, 60000);
assert.equal(receipt.physicalClosure.shardBytes, 64);
assert.equal(validateManifestConversionPostflightReceipt(receipt).ok, true);

const driftedReceipt = structuredClone(receipt);
driftedReceipt.physicalClosure.shardBytes = 65;
assert.match(
  validateManifestConversionPostflightReceipt(driftedReceipt).errors.join('; '),
  /Receipt digest does not match/
);

const corruptedShard = structuredClone(inputs.shardObservations);
corruptedShard[0].digest = stableDigest('corrupt');
assert.throws(
  () => createManifestConversionPostflightReceipt({ ...inputs, shardObservations: corruptedShard }),
  /Shard 0 digest does not match/
);

const missingTiming = structuredClone(conversionReport);
delete missingTiming.startedAtUtc;
delete missingTiming.completedAtUtc;
delete missingTiming.durationMs;
assert.throws(
  () => createManifestConversionPostflightReceipt({ ...inputs, conversionReport: missingTiming }),
  /measured physical conversion timing/
);

const checkedIn = JSON.parse(await fs.readFile(
  'reports/model-ir-v2/glimmer-30b.conversion-postflight.json',
  'utf8'
));
assert.equal(validateManifestConversionPostflightReceipt(checkedIn).ok, true);
assert.equal(checkedIn.conversionEvidence.tensorCount, 627);
assert.equal(checkedIn.physicalClosure.shardCount, 831);
assert.equal(checkedIn.dispositions.packEligible, false);

const familyNames = await fs.readFile('src/converter/manifest-conversion-postflight.js', 'utf8');
assert.doesNotMatch(familyNames, /glimmer|qwen/i, 'generic postflight must not contain model-family names');

console.log('✔ manifest-conversion-postflight.test.js passed');
