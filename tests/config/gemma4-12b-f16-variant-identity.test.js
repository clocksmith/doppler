import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const AF32_MODEL_ID = 'gemma-4-12b-it-text-q4k-ehf16-af32';
const AF16_MODEL_ID = 'gemma-4-12b-it-text-q4k-ehf16-af16';

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

const af32Config = readJson(`src/config/conversion/gemma4/${AF32_MODEL_ID}.json`);
const af16Config = readJson(`src/config/conversion/gemma4/${AF16_MODEL_ID}.json`);
const af32ManifestPath = path.join('models', 'local', AF32_MODEL_ID, 'manifest.json');
const af16ManifestPath = path.join('models', 'local', AF16_MODEL_ID, 'manifest.json');
const af32ManifestText = fs.readFileSync(af32ManifestPath, 'utf8');
const af32Manifest = JSON.parse(af32ManifestText);
const af16Manifest = readJson(af16ManifestPath);

assert.notEqual(
  fs.statSync(af32ManifestPath).ino,
  fs.statSync(af16ManifestPath).ino,
  'Gemma 4 12B af32 and af16 manifest files must not be hardlinked'
);

assert.equal(af32Config.output?.modelBaseId, AF32_MODEL_ID);
assert.equal(af16Config.output?.modelBaseId, AF16_MODEL_ID);

assert.equal(af32Manifest.modelId, AF32_MODEL_ID);
assert.equal(af16Manifest.modelId, AF16_MODEL_ID);
assert.equal(af32Manifest.quantizationInfo?.compute, 'f32');
assert.equal(af32Manifest.quantizationInfo?.variantTag, 'q4k-ehf16-af32');
assert.equal(af16Manifest.quantizationInfo?.compute, 'f16');
assert.equal(af16Manifest.quantizationInfo?.variantTag, 'q4k-ehf16-af16');

assert.equal(
  af16Manifest.artifactIdentity?.weightPackId,
  af32Manifest.artifactIdentity?.weightPackId,
  'af16 manifest variant must share the af32 weight pack id'
);
assert.equal(
  af16Manifest.artifactIdentity?.weightPackHash,
  af32Manifest.artifactIdentity?.weightPackHash,
  'af16 manifest variant must share the af32 weight pack hash'
);
assert.equal(
  af16Manifest.artifactIdentity?.shardSetHash,
  af32Manifest.artifactIdentity?.shardSetHash,
  'af16 manifest variant must share the af32 shard set'
);
assert.notEqual(
  af16Manifest.artifactIdentity?.manifestVariantId,
  af32Manifest.artifactIdentity?.manifestVariantId,
  'af16 manifest variant must carry its own manifest variant id'
);
assert.equal(af32Manifest.artifactIdentity?.artifactCompleteness, 'complete');
assert.equal(af16Manifest.artifactIdentity?.artifactCompleteness, 'weights-ref');
assert.equal(af32Manifest.weightsRef, undefined);

assert.deepEqual(af16Manifest.weightsRef, af16Config.manifest?.weightsRef);
assert.equal(af16Manifest.weightsRef?.artifactRoot, `../${AF32_MODEL_ID}`);
assert.equal(af16Manifest.weightsRef?.manifestDigest, hashText(af32ManifestText));
assert.equal(af16Manifest.weightsRef?.shardSetHash, af32Manifest.artifactIdentity?.shardSetHash);
assert.equal(af32Manifest.artifactIdentity?.conversionConfigDigest, hashJson(af32Config));
assert.equal(af16Manifest.artifactIdentity?.conversionConfigDigest, hashJson(af16Config));

assert.deepEqual(
  af32Manifest.inference?.session?.compute?.defaults,
  {
    activationDtype: 'f32',
    mathDtype: 'f32',
    accumDtype: 'f32',
    outputDtype: 'f32',
  },
  'af32 manifest session defaults must remain the f32 compute lane'
);
assert.deepEqual(
  af16Manifest.inference?.session?.compute?.defaults,
  {
    activationDtype: 'f16',
    mathDtype: 'f16',
    accumDtype: 'f16',
    outputDtype: 'f16',
  },
  'af16 manifest session defaults must make the f16 compute lane explicit'
);
assert.equal(af32Manifest.inference?.session?.perLayerInputs?.rowCache?.decodedDtype, 'f32');
assert.equal(af32Manifest.inference?.session?.perLayerInputs?.hotCache?.outputDtype, 'f32');
assert.equal(af16Manifest.inference?.session?.perLayerInputs?.rowCache?.decodedDtype, 'f16');
assert.equal(af16Manifest.inference?.session?.perLayerInputs?.hotCache?.outputDtype, 'f16');

console.log('gemma4-12b-f16-variant-identity.test: ok');
