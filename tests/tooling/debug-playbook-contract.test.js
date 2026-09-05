import assert from 'node:assert/strict';
import fs from 'node:fs';

function read(path) {
  return fs.readFileSync(path, 'utf8');
}

const agents = read('AGENTS.md');
const protocol = read('docs/agents/debug-protocol.md');
const skill = read('skills/doppler-debug/SKILL.md');
const playbook = read('docs/debug-playbook.md');
const template = read('docs/debug-investigation-template.md');

assert.match(agents, /docs\/agents\/debug-protocol\.md/);
assert.match(protocol, /docs\/debug-playbook\.md/);
assert.match(protocol, /docs\/debug-investigation-template\.md/);
assert.match(protocol, /post input norm/);
assert.match(protocol, /successful process exit/);
assert.match(protocol, /F16 or source-precision control/);

assert.match(skill, /docs\/debug-playbook\.md/);
assert.match(skill, /docs\/debug-investigation-template\.md/);
assert.match(skill, /post input norm/);
assert.match(skill, /F16 or source-precision control/);

assert.match(playbook, /tokenization \/ chat-template/);
assert.match(playbook, /conversion \/ artifact integrity/);
assert.match(playbook, /runtime numerics/);
assert.match(playbook, /surface \/ harness parity/);
assert.match(playbook, /benchmark-only/);
assert.match(playbook, /post input norm/);
assert.match(playbook, /Q\/K\/V pre-RoPE/);
assert.match(playbook, /source-precision or F16 control/i);
assert.match(playbook, /debug-investigation-template\.md/);

assert.match(template, /^# Debug Investigation Template/m);
assert.match(template, /^## Trusted reference/m);
assert.match(template, /^## Boundary diff/m);
assert.match(template, /^## First divergent boundary/m);
assert.match(template, /^## Binary split/m);
assert.match(template, /^## Conversion status/m);

console.log('debug-playbook-contract.test: ok');
