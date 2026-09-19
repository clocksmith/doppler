import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runNodeTestScripts } from '../../tools/lib/node-test-command-chain.js';
const commands = [];
assert.throws(() => runNodeTestScripts(['inventory', 'correctness'], () => {}, {
  scripts: { inventory: 'node stale-inventory.js && node dependent.js', correctness: 'node correctness.js' },
  execute(command) { commands.push(command); return { status: command.includes('stale') ? 1 : 0 }; },
}), AggregateError);
assert.deepEqual(commands, ['node stale-inventory.js', 'node correctness.js']);
const workflow = readFileSync(new URL('../../.github/workflows/check-green.yml', import.meta.url), 'utf8');
for (const script of ['test:ci', 'kernels:check', 'check:green', 'test:gpu:browser', 'test:demo:contract']) {
  const step = workflow.split(/\n      - name:/).find(block => block.includes(`run: npm run ${script}\n`));
  assert(step?.includes('!cancelled()'), `${script} must report after an independent check fails`);
  assert(step?.includes(".outcome == 'success'"), `${script} still requires successful setup`);
}
