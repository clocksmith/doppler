import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const AF32_MODEL_ID = 'gemma-4-e2b-it-q4k-ehf16-af32-int4ple';
const AF16_MODEL_ID = 'gemma-4-e2b-it-q4k-ehf16-af16-int4ple';

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function stripUndefined(value) {
  if (Array.isArray(value)) {
    return value.map(stripUndefined);
  }
  if (!value || typeof value !== 'object') {
    return value;
  }
  const output = {};
  for (const key of Object.keys(value).sort()) {
    if (value[key] !== undefined) {
      output[key] = stripUndefined(value[key]);
    }
  }
  return output;
}

function hashJson(value) {
  return `sha256:${crypto.createHash('sha256').update(JSON.stringify(stripUndefined(value))).digest('hex')}`;
}

function hashText(value) {
  return `sha256:${crypto.createHash('sha256').update(String(value)).digest('hex')}`;
}

const af16Config = readJson(`src/config/conversion/gemma4/${AF16_MODEL_ID}.json`);
const af32ManifestPath = path.join('models', 'local', AF32_MODEL_ID, 'manifest.json');
const af16ManifestPath = path.join('models', 'local', AF16_MODEL_ID, 'manifest.json');
const af32ManifestText = fs.readFileSync(af32ManifestPath, 'utf8');
const af32Manifest = JSON.parse(af32ManifestText);
const af16Manifest = readJson(af16ManifestPath);

assert.equal(af16Config.output?.modelBaseId, AF16_MODEL_ID);
assert.equal(af16Manifest.quantizationInfo?.compute, 'f16');
assert.equal(af16Manifest.quantizationInfo?.variantTag, 'q4k-ehf16-af16-vf16-audiof16-pf16-int4ple');

assert.equal(
  af16Manifest.artifactIdentity?.weightPackId,
  af32Manifest.artifactIdentity?.weightPackId,
  'af16 INT4-PLE manifest must reuse the af32 INT4-PLE weight pack'
);
assert.equal(
  af16Manifest.artifactIdentity?.shardSetHash,
  af32Manifest.artifactIdentity?.shardSetHash,
  'af16 INT4-PLE manifest must reuse the af32 INT4-PLE shard set'
);
assert.notEqual(
  af16Manifest.artifactIdentity?.manifestVariantId,
  af32Manifest.artifactIdentity?.manifestVariantId,
  'af16 INT4-PLE manifest must keep its own manifest variant id'
);

assert.deepEqual(af16Manifest.weightsRef, af16Config.manifest?.weightsRef);
assert.equal(af16Manifest.weightsRef?.artifactRoot, `../${AF32_MODEL_ID}`);
assert.equal(af16Manifest.weightsRef?.manifestDigest, hashText(af32ManifestText));
assert.equal(af16Manifest.weightsRef?.shardSetHash, af32Manifest.artifactIdentity?.shardSetHash);
assert.equal(af16Manifest.artifactIdentity?.conversionConfigDigest, hashJson(af16Config));
assert.equal(af16Manifest.conversion?.conversionConfigDigest, hashJson(af16Config));

assert.deepEqual(
  af16Manifest.inference?.session?.compute?.defaults,
  {
    activationDtype: 'f16',
    mathDtype: 'f16',
    accumDtype: 'f16',
    outputDtype: 'f16',
  },
  'af16 INT4-PLE session defaults must make the f16 lane explicit'
);
assert.equal(af16Manifest.inference?.session?.perLayerInputs?.materialization, 'range_backed');
assert.equal(af16Manifest.inference?.session?.perLayerInputs?.hotCache?.outputDtype, 'f16');

console.log('gemma4-e2b-int4ple-f16-variant-identity.test: ok');
