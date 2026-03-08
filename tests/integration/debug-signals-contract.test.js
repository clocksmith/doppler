import assert from 'node:assert/strict';

import { signalError, SIGNALS } from '../../src/debug/signals.js';

const originalConsoleLog = console.log;
const lines = [];

try {
  console.log = (line) => {
    lines.push(line);
  };

  signalError('primary failure', { code: 'E_TEST' });
  assert.equal(lines.length, 1);
  assert.match(lines[0], new RegExp(`^${SIGNALS.ERROR.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')} `));
  const payload = JSON.parse(lines[0].slice(SIGNALS.ERROR.length + 1));
  assert.equal(payload.error, 'primary failure');
  assert.equal(payload.code, 'E_TEST');

  assert.throws(
    () => signalError('primary failure', { error: 'shadowed failure' }),
    /details\.error is reserved/
  );
} finally {
  console.log = originalConsoleLog;
}

console.log('debug-signals-contract.test: ok');
