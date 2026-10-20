import assert from 'node:assert/strict';
import fs from 'node:fs';
import { detectSandwichNorm } from '../../src/inference/pipelines/text/layer.js';

function readJson(path) {
  return JSON.parse(fs.readFileSync(path, 'utf8'));
}

const config = readJson('src/config/conversion/qwen3/qwen-3-5-9b-q4k-ehaf16.json');
assert.equal(
  detectSandwichNorm(config.inference?.normalization).useSandwichNorm,
  false,
  'Qwen tensor presence must retain the standard residual-then-post-attention-norm route'
);

console.log('qwen35-9b-q4k-conversion-config-contract.test: ok');
