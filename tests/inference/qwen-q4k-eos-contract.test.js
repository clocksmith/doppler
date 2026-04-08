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
  Number.isInteger(manifest.eos_token_id),
  true,
  `${manifest.modelId} must stamp a scalar qwen EOS token id`
);

assert.equal(
  manifest.eos_token_id,
  248044,
  `${manifest.modelId} must preserve the checked-in scalar EOS token id`
);

console.log('qwen-q4k-eos-contract.test: ok');
