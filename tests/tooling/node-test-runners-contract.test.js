import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

function run(args) {
  return spawnSync(process.execPath, args, {
    cwd: process.cwd(),
    encoding: 'utf8',
  });
}

{
  const missingDir = path.join(tmpdir(), 'doppler-missing-tests-path');
  const result = run([
    'tools/run-node-tests.js',
    missingDir,
  ]);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Test path not found/);
}

{
  const missingDir = path.join(tmpdir(), 'doppler-missing-coverage-path');
  const result = run([
    'tools/run-node-coverage.js',
    missingDir,
  ]);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Test path not found/);
}

{
  const tempDir = mkdtempSync(path.join(tmpdir(), 'doppler-coverage-single-file-'));
  try {
    const testFile = path.join(tempDir, 'single.test.js');
    const policyFile = path.join(tempDir, 'coverage-policy.json');
    writeFileSync(testFile, "console.log('single.test: ok');\n", 'utf8');
    writeFileSync(policyFile, JSON.stringify({
      suite: 'all',
      thresholds: {
        line: 0,
        branch: 0,
        functions: 0,
      },
      nodeTest: {
        concurrency: 1,
        timeoutMs: 60_000,
      },
    }), 'utf8');

    const result = run([
      'tools/run-node-coverage.js',
      '--policy',
      policyFile,
      testFile,
      '--no-threshold',
    ]);
    assert.equal(result.status, 0, result.stderr || result.stdout);
    assert.match(result.stdout, /\[coverage\] all files/);
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
}

console.log('node-test-runners-contract.test: ok');
