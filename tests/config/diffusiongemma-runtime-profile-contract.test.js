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

const profileDir = path.join(repoRoot, 'src/config/runtime/profiles');
const familyModelIds = new Set(KNOWN_MODELS.map(({ modelId }) => modelId));
const profileFiles = (await fs.readdir(profileDir)).filter((name) => name.endsWith('.json'));
let checkedProfiles = 0;
for (const name of profileFiles) {
  const profile = await readJson(`src/config/runtime/profiles/${name}`);
  if (!familyModelIds.has(profile.model)) {
    continue;
  }
  checkedProfiles += 1;
  const modelId = profile.model;
  assert.equal(profile.id, `profiles/${name.slice(0, -'.json'.length)}`, `${modelId}: runtime profile id must match its path`);
  const routing = profile.runtime?.inference?.moe?.routing ?? {};
  if (routing.activeExpertSelection === 'topk-route') {
    assert.equal(
      Object.hasOwn(routing, 'maxTokensPerExpert'),
      false,
      `${modelId}: topk-route profile must not carry the readback slot budget.`
    );
  }
}
assert.ok(checkedProfiles > 0, 'DiffusionGemma must have at least one model-scoped runtime profile');

console.log('diffusiongemma-runtime-profile-contract.test: ok');
