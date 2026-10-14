import assert from 'node:assert/strict';
import fs from 'node:fs';

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

const AF32_MODEL_ID = 'gemma-3-270m-it-q4k-ehf16-af32';
const AF16_MODEL_ID = 'gemma-3-270m-it-q4k-ehf16-af16';
const AF32_CONFIG_PATH = `src/config/conversion/gemma3/${AF32_MODEL_ID}.json`;
const AF16_CONFIG_PATH = `src/config/conversion/gemma3/${AF16_MODEL_ID}.json`;

const af32Config = readJson(AF32_CONFIG_PATH);
const af16Config = readJson(AF16_CONFIG_PATH);

assert.equal(
  af16Config.manifest?.artifactIdentity?.sourceCheckpointId,
  af32Config.manifest?.artifactIdentity?.sourceCheckpointId,
  'af16 repro must use the same source checkpoint as the fair af32 lane'
);
assert.equal(
  af16Config.manifest?.artifactIdentity?.weightCapsuleId,
  af32Config.manifest?.artifactIdentity?.weightCapsuleId,
  'af16 repro must use the same Q4K weight-capsule identity as the fair af32 lane'
);
assert.notEqual(
  af16Config.manifest?.artifactIdentity?.manifestVariantId,
  af32Config.manifest?.artifactIdentity?.manifestVariantId,
  'af16 repro must carry a distinct manifest variant identity'
);

console.log('gemma3-270m-af16-conversion-config-contract.test: ok');
