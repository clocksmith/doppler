import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { KNOWN_MODELS } from '../../src/models/diffusiongemma.js';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, '..', '..');

async function readJson(relativePath) {
  const filePath = path.join(repoRoot, relativePath);
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
}

const profile = await readJson('src/config/runtime/profiles/diffusiongemma-26b-a4b-throughput.json');
const modelProfiles = new Map(KNOWN_MODELS.map((entry) => [entry.modelId, entry.defaultRuntimeProfile]));

assert.equal(
  modelProfiles.get('diffusiongemma-26b-a4b-it-q4k-ehf16-af16'),
  'profiles/diffusiongemma-26b-a4b-throughput',
  'DiffusionGemma must advertise the measured active-expert throughput profile.'
);
assert.equal(profile.id, 'profiles/diffusiongemma-26b-a4b-throughput');
assert.equal(profile.extends, 'profiles/throughput');
assert.equal(profile.intent, 'calibrate');
assert.equal(profile.model, 'diffusiongemma-26b-a4b-it-q4k-ehf16-af16');
assert.equal(
  profile.runtime?.inference?.moe?.routing?.activeExpertSelection,
  'topk-readback',
  'DiffusionGemma throughput profile must pin the fast active-expert scheduler.'
);
assert.equal(
  profile.runtime?.inference?.moe?.routing?.maxTokensPerExpert,
  64,
  'DiffusionGemma throughput profile must pin the measured active-expert slot budget.'
);

console.log('diffusiongemma-runtime-profile-contract.test: ok');
