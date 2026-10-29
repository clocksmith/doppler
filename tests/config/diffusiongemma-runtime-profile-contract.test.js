import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, '..', '..');

async function readJson(relativePath) {
  const filePath = path.join(repoRoot, relativePath);
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
}

const profileDir = 'src/config/runtime/profiles';
const profileNames = (await fs.readdir(path.join(repoRoot, profileDir)))
  .filter((name) => name.startsWith('diffusiongemma-') && name.endsWith('.json'));
assert.ok(profileNames.length > 0, 'expected at least one DiffusionGemma runtime profile');
for (const name of profileNames) {
  const profile = await readJson(`${profileDir}/${name}`);
  const modelId = profile.model;
  assert.equal(profile.id, `profiles/${name.slice(0, -'.json'.length)}`, `${name}: runtime profile id must match its path`);
  assert.match(modelId, /^diffusiongemma-/, `${name}: runtime profile must be scoped to a DiffusionGemma model`);
  const routing = profile.runtime?.inference?.moe?.routing ?? {};
  if (routing.activeExpertSelection === 'topk-route') {
    assert.equal(
      Object.hasOwn(routing, 'maxTokensPerExpert'),
      false,
      `${modelId}: topk-route profile must not carry the readback slot budget.`
    );
  }
}

console.log('diffusiongemma-runtime-profile-contract.test: ok');
