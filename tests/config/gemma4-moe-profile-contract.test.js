import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, '..', '..');

// === Load both MoE kernel profiles ===

const mixtralProfile = JSON.parse(
  await fs.readFile(path.join(repoRoot, 'src/config/kernels/moe/mixtral.paths.json'), 'utf8')
);
const gptOssProfile = JSON.parse(
  await fs.readFile(path.join(repoRoot, 'src/config/kernels/moe/gpt-oss.paths.json'), 'utf8')
);

// === Mixtral profile has valid structure ===

{
  assert.equal(mixtralProfile.id, 'mixtral-moe-v1');
  assert.ok(mixtralProfile.description, 'must have description');
}

// === Router rules are complete ===

{
  const router = mixtralProfile.router;
  assert.ok(router, 'must define router');
  assert.ok(router.topk, 'must define router.topk rules');
  assert.ok(Array.isArray(router.topk), 'router.topk must be an array');
  assert.ok(router.topk.length >= 2, 'router.topk must have at least 2 rules');

  // Every rule must have match and value
  for (const rule of router.topk) {
    assert.ok(rule.match !== undefined, 'each rule must have match');
    assert.ok(typeof rule.value === 'string' && rule.value.length > 0,
      'each rule must have a non-empty string value');
  }

  // Last rule must be a catch-all (empty match)
  const lastRule = router.topk[router.topk.length - 1];
  assert.deepEqual(lastRule.match, {},
    'last router rule must be a catch-all with empty match');
}

// === Dequant rules are complete ===

{
  const dequant = mixtralProfile.dequant;
  assert.ok(dequant, 'must define dequant');

  // Must have Q4K expert dequant rules
  assert.ok(dequant.q4kExpert, 'must define dequant.q4kExpert');
  assert.ok(Array.isArray(dequant.q4kExpert), 'dequant.q4kExpert must be an array');
  assert.ok(dequant.q4kExpert.length >= 2, 'must have at least 2 q4kExpert rules');

  // Must have F16 expert rules
  assert.ok(dequant.f16Expert, 'must define dequant.f16Expert');
  assert.ok(Array.isArray(dequant.f16Expert), 'dequant.f16Expert must be an array');

  // Every dequant rule array must end with a catch-all
  for (const [name, rules] of Object.entries(dequant)) {
    const lastRule = rules[rules.length - 1];
    assert.deepEqual(lastRule.match, {},
      `dequant.${name} last rule must be a catch-all`);
  }
}

// === Structural parity with GPT-OSS profile ===

{
  // Both profiles must have the same top-level structure
  assert.ok(gptOssProfile.id, 'gpt-oss profile must have id');
  assert.ok(gptOssProfile.router, 'gpt-oss profile must have router');
  assert.ok(gptOssProfile.dequant, 'gpt-oss profile must have dequant');

  // Both must have router.topk
  assert.ok(gptOssProfile.router.topk, 'gpt-oss must have router.topk');
  assert.ok(mixtralProfile.router.topk, 'mixtral must have router.topk');

  // Both router.topk must be rule arrays with match/value
  for (const rule of gptOssProfile.router.topk) {
    assert.ok(rule.match !== undefined && rule.value, 'gpt-oss router rules must have match+value');
  }
}

// === Mixtral profile does not reference GPT-OSS-specific formats ===

{
  const mixtralJson = JSON.stringify(mixtralProfile);
  assert.ok(!mixtralJson.includes('mxfp4'),
    'mixtral profile must not reference MXFP4 (GPT-OSS-specific format)');
  assert.ok(!mixtralJson.includes('gptoss'),
    'mixtral profile must not reference gptoss variants');
}

console.log('gemma4-moe-profile-contract.test: ok');
