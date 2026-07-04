import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { resolveTestFiles } from '../../tools/run-node-coverage.js';

const suiteDir = mkdtempSync(path.join(tmpdir(), 'doppler-coverage-pending-'));

try {
  const keptTest = path.join(suiteDir, 'kept.test.js');
  const pendingTest = path.join(suiteDir, 'future.pending.test.js');

  writeFileSync(keptTest, "console.log('coverage-kept-contract: ok');\n");
  writeFileSync(pendingTest, "throw new Error('pending coverage test ran');\n");

  assert.deepEqual(resolveTestFiles('all', [suiteDir]), [keptTest]);
  assert.deepEqual(resolveTestFiles('all', [pendingTest]), [pendingTest]);
} finally {
  rmSync(suiteDir, { recursive: true, force: true });
}

console.log('node-coverage-pending-contract.test: ok');
