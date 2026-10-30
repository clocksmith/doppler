import assert from 'node:assert/strict';
import { PipelineGenerator } from '../../src/inference/pipelines/text/generator.js';
const generator = new PipelineGenerator({ tokenizer: { getSpecialTokens: () => ({ pad: 2 }) }, modelConfig: { vocabSize: 4 } });
const input = Float32Array.of(1, 3, 100, 2), observations = [];
const options = { repetitionPenalty: 1, repetitionPenaltyWindow: 0, presencePenalty: 0,
  temperature: 0, topP: 1, topK: 1, suppressTokenIds: [1],
  onLogits: (values, metadata) => observations.push({ values: Array.from(values), metadata }) };
assert.equal(generator._sampleNextTokenFromLogits(input, [], options), 3);
assert.deepEqual(observations[0].values, [1, 3, 100, 2], 'padding and token suppression must not corrupt observed numerical vectors');
assert.equal(observations[0].metadata.tokenId, 3);
assert.deepEqual(Array.from(input), [1, 3, 100, 2]);
assert.equal(generator._sampleNextTokenFromLogits(input, [], { ...options, onLogits: null }), 3);
console.log('generator-prefill-logits-observation.test: ok');
