import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, '..', '..');
const profilePath = path.join(
  repoRoot,
  'src/config/runtime/experiments/debug/gemma3-debug-q4k.json'
);

const profile = JSON.parse(await fs.readFile(profilePath, 'utf8'));

assert.equal(profile.id, 'experiments/debug/gemma3-debug-q4k');
assert.equal(profile.extends, 'default');
assert.equal(profile.runtime.inference.prompt, 'Hello');
assert.deepEqual(
  profile.runtime.inference.sampling,
  {
    temperature: 0,
    topK: 1,
    topP: 1,
  },
  'Gemma 3 Q4K debug profile must use deterministic sampling so debug coherence does not depend on random decode choices.'
);

console.log('gemma3-debug-profile-contract.test: ok');
