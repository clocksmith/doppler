import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const CASES = [
  'models/local/gemma-3-1b-it-f16-af32',
  'models/local/gemma-3-1b-it-q4k-ehf16-af32',
  'models/local/gemma-3-270m-it-f16',
  'models/local/gemma-3-270m-it-f16-af32',
];

for (const modelDir of CASES) {
  const manifest = JSON.parse(fs.readFileSync(path.join(modelDir, 'manifest.json'), 'utf8'));
  const tokenizer = JSON.parse(fs.readFileSync(path.join(modelDir, 'tokenizer.json'), 'utf8'));
  const actualVocabSize = Array.isArray(tokenizer?.model?.vocab)
    ? tokenizer.model.vocab.length
    : Object.keys(tokenizer?.model?.vocab || {}).length;

  assert.equal(
    manifest?.tokenizer?.vocabSize,
    actualVocabSize,
    `${modelDir} manifest.tokenizer.vocabSize must match tokenizer.json vocab size`
  );
}

console.log('local-gemma-tokenizer-vocab-contract.test: ok');
