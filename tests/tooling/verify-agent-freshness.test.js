import assert from 'node:assert/strict';
import fs from 'node:fs';

import { REQUIRED_SKILL_SYMLINKS } from '../../tools/verify-agent-freshness.js';

assert.deepEqual(REQUIRED_SKILL_SYMLINKS, [
  ['.claude/skills', '../skills'],
  ['.gemini/skills', '../skills'],
  ['.codex/skills', '../skills'],
]);

{
  const policy = JSON.parse(fs.readFileSync('tools/policies/agent-parity-policy.json', 'utf8'));
  assert.deepEqual(policy.skillAliases, [
    { path: '.claude/skills', target: '../skills' },
    { path: '.gemini/skills', target: '../skills' },
    { path: '.codex/skills', target: '../skills' },
  ]);
}

console.log('verify-agent-freshness.test: ok');
