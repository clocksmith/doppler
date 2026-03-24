import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';

const manifestPath = new URL(
  '../../models/local/qwen-3-5-0-8b-q4k-ehaf16/manifest.json',
  import.meta.url
);

if (!existsSync(manifestPath)) {
  console.log('qwen-q4k-eos-contract.test: skipped (local q4k fixture missing)');
  process.exit(0);
}

const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));

assert.equal(
  Array.isArray(manifest.eos_token_id),
  true,
  `${manifest.modelId} must stamp qwen stop tokens as an explicit array`
);

assert.deepEqual(
  manifest.eos_token_id,
  [248046, 248069],
  `${manifest.modelId} must stop on <|im_end|> first and </think> second`
);

console.log('qwen-q4k-eos-contract.test: ok');
