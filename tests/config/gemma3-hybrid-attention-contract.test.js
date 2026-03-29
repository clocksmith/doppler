import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

function readJson(filePath) {
  return JSON.parse(readFileSync(filePath, 'utf8'));
}

const gemma3Configs = [
  'src/config/conversion/gemma3/gemma-3-270m-it-q4k-ehf16-af32.json',
  'src/config/conversion/gemma3/gemma-3-1b-it-f16-af32.json',
  'src/config/conversion/gemma3/gemma-3-1b-it-q4k-ehf16-af32.json',
  'src/config/conversion/gemma3/translategemma-4b-it-q4k-ehf16-af32.json',
  'src/config/conversion/gemma3/translategemma-4b-1b-enes-q4k-ehf16-af32.json',
];

for (const filePath of gemma3Configs) {
  const config = readJson(filePath);
  const inference = config.inference ?? {};
  assert.equal(inference.attention?.slidingWindow, 512, `${filePath}: slidingWindow`);
  assert.equal(inference.rope?.ropeLocalTheta, 10000, `${filePath}: ropeLocalTheta`);
  assert.equal(inference.layerPattern?.type, 'every_n', `${filePath}: layerPattern.type`);
  assert.equal(inference.layerPattern?.period, 6, `${filePath}: layerPattern.period`);
  assert.equal(inference.layerPattern?.offset, null, `${filePath}: layerPattern.offset`);
}

console.log('gemma3-hybrid-attention-contract.test: ok');
