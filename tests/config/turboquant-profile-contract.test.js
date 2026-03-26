import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, '..', '..');
const profilesDir = path.join(repoRoot, 'src/config/runtime/profiles');

const PROFILE_FILES = [
  'turboquant.json',
  'turboquant-contiguous.json',
  'turboquant-contiguous-prod.json',
];

const profiles = {};
for (const file of PROFILE_FILES) {
  const filePath = path.join(profilesDir, file);
  profiles[file] = JSON.parse(await fs.readFile(filePath, 'utf8'));
}

for (const [file, profile] of Object.entries(profiles)) {
  assert.ok(profile.id.startsWith('profiles/'), `${file}: id must be in the profiles namespace`);
  assert.equal(profile.extends, 'profiles/throughput', `${file}: must extend profiles/throughput`);
  assert.ok(profile.runtime, `${file}: must use top-level runtime payload`);
  assert.equal(profile.inference, undefined, `${file}: must not use legacy top-level inference`);

  const sessionKV = profile.runtime.inference?.session?.kvcache;
  assert.ok(sessionKV, `${file}: must define runtime.inference.session.kvcache`);
  assert.equal(profile.runtime.inference?.kvcache, undefined, `${file}: must not mirror session.kvcache to runtime.inference.kvcache`);
  assert.equal(sessionKV.maxSeqLen, 2048, `${file}: maxSeqLen must match current TurboQuant kernel ceiling`);
  assert.equal(sessionKV.kvDtype, 'f16', `${file}: kvDtype must be f16`);
}

{
  const tiered = profiles['turboquant.json'];
  assert.equal(tiered.runtime.inference.session.kvcache.layout, 'tiered');
  assert.equal(tiered.runtime.inference.session.kvcache.tiering.mode, 'turboquant');
  assert.equal(tiered.runtime.inference.session.kvcache.tiering.compression.mode, 'turboquant');
}

{
  const contiguous = profiles['turboquant-contiguous.json'];
  assert.equal(contiguous.runtime.inference.session.kvcache.layout, 'contiguous');
  assert.equal(contiguous.runtime.inference.session.kvcache.tiering.mode, 'off');
  assert.equal(contiguous.runtime.inference.session.kvcache.quantization.mode, 'turboquant');
  assert.equal(contiguous.runtime.inference.session.kvcache.quantization.prodMode, false);
}

{
  const prod = profiles['turboquant-contiguous-prod.json'];
  assert.equal(prod.runtime.inference.session.kvcache.layout, 'contiguous');
  assert.equal(prod.runtime.inference.session.kvcache.tiering.mode, 'off');
  assert.equal(prod.runtime.inference.session.kvcache.quantization.mode, 'turboquant_prod');
  assert.equal(prod.runtime.inference.session.kvcache.quantization.prodMode, true);
}

console.log('turboquant-profile-contract.test: ok');
