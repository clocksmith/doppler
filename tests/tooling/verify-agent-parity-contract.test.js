import assert from 'node:assert/strict';
import fs from 'node:fs';

import {
  REQUIRED_INSTRUCTION_ALIASES,
  REQUIRED_SKILL_ALIASES,
} from '../../tools/verify-agent-parity.js';

assert.deepEqual(REQUIRED_INSTRUCTION_ALIASES, [
  { path: 'CLAUDE.md', target: 'AGENTS.md' },
  { path: 'GEMINI.md', target: 'AGENTS.md' },
]);
assert.deepEqual(REQUIRED_SKILL_ALIASES, [
  { path: '.claude/skills', target: '../skills' },
  { path: '.gemini/skills', target: '../skills' },
  { path: '.codex/skills', target: '../skills' },
]);

{
  const policy = JSON.parse(fs.readFileSync('tools/policies/agent-parity-policy.json', 'utf8'));
  for (const alias of REQUIRED_INSTRUCTION_ALIASES) {
    assert.equal(
      policy.instructionAliases.some((entry) => entry.path === alias.path && entry.target === alias.target),
      true
    );
  }
  for (const alias of REQUIRED_SKILL_ALIASES) {
    assert.equal(
      policy.skillAliases.some((entry) => entry.path === alias.path && entry.target === alias.target),
      true
    );
  }
}

console.log('verify-agent-parity-contract.test: ok');
