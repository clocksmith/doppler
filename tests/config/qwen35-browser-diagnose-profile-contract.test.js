import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, '..', '..');
const profilePath = path.join(
  repoRoot,
  'src/config/runtime/experiments/debug/qwen35-browser-diagnose-q4k.json'
);

const profile = JSON.parse(await fs.readFile(profilePath, 'utf8'));

assert.equal(profile.id, 'experiments/debug/qwen35-browser-diagnose-q4k');
assert.equal(profile.extends, 'profiles/default');
assert.equal(profile.model, 'qwen-3-5-0-8b-q4k-ehaf16');
assert.equal(profile.runtime.shared.tooling.intent, 'investigate');
assert.equal(profile.runtime.shared.tooling.diagnostics, 'always');
assert.equal(profile.runtime.shared.debug.profiler.enabled, true);
assert.equal(profile.runtime.inference.chatTemplate.enabled, false);
assert.equal(profile.runtime.inference.generation.maxTokens, 1);
assert.deepEqual(
  profile.runtime.inference.sampling,
  {
    temperature: 0,
    topK: 1,
    topP: 1,
  },
  'Qwen 3.5 browser diagnosis profile must keep deterministic compare-lane sampling.'
);
assert.deepEqual(
  profile.runtime.inference.kernelPathPolicy,
  {
    mode: 'capability-aware',
    sourceScope: ['model', 'manifest', 'config'],
    allowSources: ['model', 'manifest', 'config'],
    onIncompatible: 'remap',
  },
  'Qwen 3.5 browser diagnosis profile must preserve capability-aware compare-lane remap policy.'
);

console.log('qwen35-browser-diagnose-profile-contract.test: ok');
