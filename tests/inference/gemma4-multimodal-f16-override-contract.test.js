import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const generatorSource = readFileSync(new URL('../../src/inference/pipelines/text/generator.js', import.meta.url), 'utf8');

assert.doesNotMatch(
  generatorSource,
  /embedding overrides currently require f32 activations/,
  'Gemma 4 multimodal prefix overrides must not hard-fail on f16 activations'
);

assert.match(
  generatorSource,
  /baseTensor\.dtype !== 'f32' && baseTensor\.dtype !== 'f16'/,
  'Gemma 4 multimodal prefix overrides must accept both f32 and f16 embedding tensors'
);

assert.match(
  generatorSource,
  /castF32ToF16\(overrideTensor\)|f32ToF16Array\(override\.embeddings\)/,
  'Gemma 4 multimodal prefix overrides must convert visual features into f16 when the text prefill path runs f16'
);

assert.match(
  generatorSource,
  /assertImplicitDtypeTransitionAllowed\(\{[\s\S]*op: 'prefill_embedding_override'/,
  'Gemma 4 multimodal prefix overrides must fail fast when execution-v1 requires explicit cast steps'
);

console.log('gemma4-multimodal-f16-override-contract.test: ok');
