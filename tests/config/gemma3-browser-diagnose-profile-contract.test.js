import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, '..', '..');
const profilePath = path.join(
  repoRoot,
  'src/config/runtime/experiments/debug/gemma3-browser-diagnose-q4k.json'
);

const profile = JSON.parse(await fs.readFile(profilePath, 'utf8'));

assert.equal(profile.id, 'experiments/debug/gemma3-browser-diagnose-q4k');
assert.equal(profile.extends, 'profiles/default');
assert.equal(profile.model, 'gemma-3-1b-it-q4k-ehf16-af32');
assert.equal(profile.runtime.shared.tooling.intent, 'investigate');
assert.equal(profile.runtime.shared.tooling.diagnostics, 'always');
assert.equal(profile.runtime.shared.debug.profiler.enabled, true);
assert.equal(profile.runtime.inference.chatTemplate.enabled, false);
assert.equal(profile.runtime.inference.generation.maxTokens, 64);
assert.deepEqual(
  profile.runtime.inference.sampling,
  {
    temperature: 0,
    topK: 1,
    topP: 1,
  },
  'Gemma 3 browser diagnosis profile must keep deterministic compare-lane sampling.'
);
assert.deepEqual(
  profile.runtime.inference.kernelPathPolicy,
  {
    mode: 'capability-aware',
    sourceScope: ['model', 'manifest', 'config'],
    allowSources: ['model', 'manifest', 'config'],
    onIncompatible: 'remap',
  },
  'Gemma 3 browser diagnosis profile must preserve capability-aware compare-lane remap policy.'
);

console.log('gemma3-browser-diagnose-profile-contract.test: ok');
