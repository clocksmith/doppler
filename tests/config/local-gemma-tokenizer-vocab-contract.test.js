import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const RDRR_DIRS = [
  'models/local',
  path.join(process.env.DOPPLER_EXTERNAL_MODELS_ROOT || '/media/x/models', 'rdrr'),
];

function discoverModels(roots) {
  const dirs = [];
  for (const root of roots) {
    if (!fs.existsSync(root)) continue;
    for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const dir = path.join(root, entry.name);
      if (fs.existsSync(path.join(dir, 'manifest.json')) && fs.existsSync(path.join(dir, 'tokenizer.json'))) {
        dirs.push(dir);
      }
    }
  }
  return dirs;
}

const cases = discoverModels(RDRR_DIRS);

if (cases.length === 0) {
  console.log('local-gemma-tokenizer-vocab-contract.test: skipped (no RDRR artifacts with manifest + tokenizer found)');
  process.exit(0);
}

let verified = 0;

for (const modelDir of cases) {
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
  verified++;
}

console.log(`local-gemma-tokenizer-vocab-contract.test: ok (${verified} verified across ${RDRR_DIRS.filter((d) => fs.existsSync(d)).length} roots)`);
