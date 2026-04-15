import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const generatorStepsSource = readFileSync(
  new URL('../../src/inference/pipelines/text/generator-steps.js', import.meta.url),
  'utf8'
);

assert.match(
  generatorStepsSource,
  /currentTokenIdsArray\[0\][\s\S]*?rollingIds\[i\]/,
  'GPU batched decode must update the per-layer token context with the current rolling token index.'
);

assert.match(
  generatorStepsSource,
  /tokenIdHint:\s*Number\.isInteger\(rollingIds\[i\]\)\s*\?\s*rollingIds\[i\]\s*:\s*null/,
  'preparePerLayerInputs tokenIdHint should use the current rolling token when available.'
);

console.log('generator-token-id-hint-contract.test: ok');
