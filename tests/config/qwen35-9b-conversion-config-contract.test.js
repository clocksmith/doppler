import assert from 'node:assert/strict';
import fs from 'node:fs';
import { detectSandwichNorm } from '../../src/inference/pipelines/text/layer.js';

const CONFIG_PATHS = [
  'src/config/conversion/qwen3/qwen-3-5-9b-f16-af32.json',
  'src/config/conversion/qwen3/qwen-3-5-9b-q4k-ehaf16.json',
];

for (const path of CONFIG_PATHS) {
  const config = JSON.parse(fs.readFileSync(path, 'utf8'));
  const executionJson = JSON.stringify(config.execution);

  assert.equal(
    detectSandwichNorm(config.inference?.normalization).useSandwichNorm,
    false,
    `${path}: Qwen tensor presence must retain the standard residual-then-post-attention-norm route`
  );
  if (config.quantization?.weights === 'f16') {
    assert.doesNotMatch(executionJson, /q4_/, `${path}: f16 weights must not bind Q4 kernels`);
  }
}

console.log('qwen35-9b-conversion-config-contract.test: ok');
