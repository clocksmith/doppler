import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { resolvePleHotVocabularyCapacity } from '../../src/inference/pipelines/text/per-layer-inputs.js';

const source = readFileSync(
  new URL('../../src/inference/pipelines/text/per-layer-inputs.js', import.meta.url),
  'utf8'
);

assert.match(
  source,
  /const useHotVocabularyTables = hasHotVocabularyTables && perLayerTokenIds != null;/,
  'Gemma 4 hot-vocab PLE must only route through hot tables when perLayerTokenIds are explicit hot-table indices'
);

const capacity = resolvePleHotVocabularyCapacity({
  maxTokens: 4096,
  maxBytes: 268435456,
  numLayers: 35,
  hiddenSize: 8960,
  bytesPerElement: 4,
  vocabSize: 262144,
});
assert.equal(capacity.maxHotTokens, 212);
assert.ok(
  (capacity.maxHotTokens + 1) * capacity.bytesPerHotRow + capacity.tokenIndexMapBytes <= 268435456,
  'Gemma 4 hot-vocab tables and token map must stay within maxBytes'
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

assert.match(
  source,
  /const expandsF16ToF32 = sourceDtype === 'f16' && policy\.outputDtype === 'f32';/,
  'Gemma 4 AF32 hot-vocab PLE must accept its range-backed f16 source rows'
);

assert.match(
  source,
  /expandedF32Row\[valueIdx\] = f16Lookup\[sourceWords\[valueIdx\]\];/,
  'Gemma 4 AF32 hot-vocab PLE must expand f16 source values through the shared lookup table'
);

console.log('gemma4-hot-vocab-ple-contract.test: ok');
