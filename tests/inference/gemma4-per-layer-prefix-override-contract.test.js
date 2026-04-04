import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const generatorSource = readFileSync(new URL('../../src/inference/pipelines/text/generator.js', import.meta.url), 'utf8');

assert.match(
  generatorSource,
  /const embeddingInputIds = resolvePrefillEmbeddingInputIds\(\s*inputIds,\s*opts\?\.embeddingInputSpan \?\? null,\s*'_prefill'\s*\);/,
  'Gemma 4 prefill must resolve an explicit embeddingInputSpan when multimodal spans need PAD-token embeddings'
);

assert.match(
  generatorSource,
  /preparePerLayerInputs\(\s*embeddingInputIds,\s*embeddingInputIds === inputIds \? hiddenStates : baseEmbeddings,\s*context,\s*\{/,
  'Gemma 4 per-layer input preparation must use the PAD-token base embedding view when multimodal embeddingInputIds are supplied'
);

console.log('gemma4-per-layer-prefix-override-contract.test: ok');
