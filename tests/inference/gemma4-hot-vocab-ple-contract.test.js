import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const source = readFileSync(
  new URL('../../src/inference/pipelines/text/per-layer-inputs.js', import.meta.url),
  'utf8'
);

assert.match(
  source,
  /const useHotVocabularyTables = hasHotVocabularyTables && perLayerTokenIds != null;/,
  'Gemma 4 hot-vocab PLE must only route through hot tables when perLayerTokenIds are explicit hot-table indices'
);

assert.match(
  source,
  /const layerEmbedSource = useHotVocabularyTables\s+\? getEmbeddingSource\(hotVocabularyRuntime\.splitTables\[layerIdx\], `embedTokensPerLayerHot\[L\$\{layerIdx\}\]`\)\s+: hasSplitEmbeddingTables/,
  'Gemma 4 hot-vocab PLE must not read from hot split tables for non-hot tokens'
);

assert.match(
  source,
  /vocabSize: useHotVocabularyTables \? \(hotVocabularyRuntime\.sentinelIndex \+ 1\) : vocabSizePerLayerInput,/,
  'Gemma 4 hot-vocab PLE must use the hot-table vocab size only on the hot-token path'
);

console.log('gemma4-hot-vocab-ple-contract.test: ok');
