import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

const tempDir = mkdtempSync(path.join(tmpdir(), 'doppler-run-node-tests-'));
try {
  writeFileSync(
    path.join(tempDir, 'a-first.test.js'),
    "globalThis.__dopplerIsolationLeak = 1;\nconsole.log('a-first.test: ok');\n",
    'utf8'
  );
  writeFileSync(
    path.join(tempDir, 'b-second.test.js'),
    [
      "if (globalThis.__dopplerIsolationLeak !== undefined) {",
      "  throw new Error('test process state leaked across files');",
      '}',
      "console.log('b-second.test: ok');",
      '',
    ].join('\n'),
    'utf8'
  );

  const result = spawnSync(process.execPath, [
    'tools/run-node-tests.mjs',
    tempDir,
    '--force-exit',
  ], {
    cwd: process.cwd(),
    encoding: 'utf8',
  });

  assert.equal(result.status, 0, result.stderr || result.stdout);
  assert.match(result.stdout, /\[node-tests\] ok: 2 files/);
} finally {
  rmSync(tempDir, { recursive: true, force: true });
}

console.log('run-node-tests-isolation.test: ok');
