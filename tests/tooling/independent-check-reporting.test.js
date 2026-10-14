import assert from 'node:assert/strict';
import { runNodeTestScripts } from '../../tools/lib/node-test-command-chain.js';
const commands = [];
assert.throws(() => runNodeTestScripts(['inventory', 'correctness'], () => {}, {
  scripts: { inventory: 'node stale-inventory.js && node dependent.js', correctness: 'node correctness.js' },
  execute(command) { commands.push(command); return { status: command.includes('stale') ? 1 : 0 }; },
}), AggregateError);
assert.deepEqual(commands, ['node stale-inventory.js', 'node correctness.js']);
