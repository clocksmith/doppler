// Historical artifact identities are checked against their frozen conversion inputs.
// Current Capsule configurations require new conversion/qualification, not relabeling.
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';

const MODEL_ID = 'gemma-4-31b-it-text-q4k-ehf16-af32';
const CONFIG_PATH = `tests/fixtures/pre-capsule/src/config/conversion/gemma4/${MODEL_ID}.json`;
const MANIFEST_PATH = `models/local/${MODEL_ID}/manifest.json`;

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

const conversionConfig = readJson(CONFIG_PATH);

if (fs.existsSync(MANIFEST_PATH)) {
  const manifest = readJson(MANIFEST_PATH);
  const numLayers = manifest.architecture?.numLayers;
  assert.ok(Number.isInteger(numLayers) && numLayers > 0, 'local manifest: architecture.numLayers');
  for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
    assert.ok(
      manifest.tensors?.[`model.language_model.layers.${layerIdx}.layer_scalar`],
      `local manifest: layer ${layerIdx} layer_scalar tensor must be present`
    );
  }
  assert.equal(
    manifest.artifactIdentity?.manifestVariantId,
    conversionConfig.manifest?.artifactIdentity?.manifestVariantId,
    'local manifest: manifest variant must identify its conversion config'
  );
  assert.equal(
    manifest.artifactIdentity?.conversionConfigDigest,
    hashJson(conversionConfig),
    'local manifest: conversionConfigDigest must match the frozen conversion config'
  );
} else {
  console.log(`gemma4-31b-q4-fused-path-contract.test: skipped local manifest (${MANIFEST_PATH} missing)`);
}

console.log('gemma4-31b-q4-fused-path-contract.test: ok');
