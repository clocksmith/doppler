import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const generatorStepsSource = readFileSync(
  new URL('../../src/inference/pipelines/text/generator-steps.js', import.meta.url),
  'utf8'
);

assert.match(
  generatorStepsSource,
  /currentTokenIdsArray\[0\]\s*=\s*i\s*===\s*0\s*\?\s*startToken\s*:\s*null/,
  'GPU batched decode must seed currentTokenIds with startToken only at i===0; tokens at i>0 are sampled on the GPU and unknown to the CPU until readback.'
);

assert.match(
  generatorStepsSource,
  /tokenIdHint:\s*i\s*===\s*0\s*\?\s*startToken\s*:\s*null/,
  'preparePerLayerInputs tokenIdHint must use startToken at i===0 and null otherwise so the PLE cache skips CPU-side token-dependent optimizations during batched iterations.'
);

assert.doesNotMatch(
  generatorStepsSource,
  /currentTokenIdsArray\[0\]\s*=\s*rollingIds\[/,
  'The legacy `rollingIds[i]` path must not reappear in the batched GPU decode loop — rollingIds is only defined in the stepwise fallback path, and referencing it here caused a ReferenceError that collapsed batched decode to ~10 tok/s.'
);

assert.doesNotMatch(
  generatorStepsSource,
  /tokenIdHint:\s*Number\.isInteger\(rollingIds\[/,
  'The legacy `rollingIds[i]` tokenIdHint path must not reappear in the batched GPU decode loop.'
);

console.log('generator-token-id-hint-contract.test: ok');
