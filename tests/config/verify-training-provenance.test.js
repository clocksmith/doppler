import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { closeSync, mkdtempSync, openSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

function runNodeScript(args) {
  const logDir = mkdtempSync(path.join(tmpdir(), 'doppler-provenance-run-'));
  const stdoutPath = path.join(logDir, 'stdout.log');
  const stderrPath = path.join(logDir, 'stderr.log');
  const stdoutFd = openSync(stdoutPath, 'w');
  const stderrFd = openSync(stderrPath, 'w');

  const result = spawnSync(process.execPath, args, {
    cwd: process.cwd(),
    stdio: ['ignore', stdoutFd, stderrFd],
  });

  closeSync(stdoutFd);
  closeSync(stderrFd);

  const output = {
    code: result.status ?? 1,
    stdout: readFileSync(stdoutPath, 'utf8'),
    stderr: readFileSync(stderrPath, 'utf8'),
  };
  rmSync(logDir, { recursive: true, force: true });
  return output;
}

{
  const result = runNodeScript(['tools/verify-training-provenance.mjs']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /Usage: node tools\/verify-training-provenance\.mjs --manifest/);
}

{
  const result = runNodeScript(['tools/verify-training-provenance.mjs', '--nope']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /Unknown flag: --nope/);
}

{
  const result = runNodeScript(['tools/verify-training-provenance.mjs', '--self-test']);
  assert.equal(result.code, 0);
  assert.match(result.stdout, /self-test ok/);
}

{
  const tempDir = mkdtempSync(path.join(tmpdir(), 'doppler-provenance-report-test-'));
  try {
    const reportPath = path.join(tempDir, 'report.json');
    writeFileSync(reportPath, JSON.stringify({
      suite: 'training',
      modelId: 'toy-model',
      metrics: null,
    }, null, 2), 'utf8');
    const result = runNodeScript([
      'tools/verify-training-provenance.mjs',
      '--report',
      reportPath,
    ]);
    assert.equal(result.code, 0);
    assert.match(result.stdout, /report ok/);
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
}

{
  const tempDir = mkdtempSync(path.join(tmpdir(), 'doppler-provenance-invalid-report-test-'));
  try {
    const reportPath = path.join(tempDir, 'invalid-report.json');
    writeFileSync(reportPath, JSON.stringify({ modelId: 'toy-model' }, null, 2), 'utf8');
    const result = runNodeScript([
      'tools/verify-training-provenance.mjs',
      '--report',
      reportPath,
    ]);
    assert.equal(result.code, 1);
    assert.match(result.stderr, /report\.suite must be a non-empty string\./);
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
}

{
  const tempDir = mkdtempSync(path.join(tmpdir(), 'doppler-provenance-invalid-checkpoint-test-'));
  try {
    const reportPath = path.join(tempDir, 'report.json');
    const checkpointPath = path.join(tempDir, 'checkpoint.json');
    writeFileSync(reportPath, JSON.stringify({
      suite: 'training',
      modelId: 'toy-model',
      metrics: null,
    }, null, 2), 'utf8');
    writeFileSync(checkpointPath, JSON.stringify({ metadata: {} }, null, 2), 'utf8');
    const result = runNodeScript([
      'tools/verify-training-provenance.mjs',
      '--report',
      reportPath,
      '--checkpoint',
      checkpointPath,
    ]);
    assert.equal(result.code, 1);
    assert.match(result.stderr, /checkpoint\.metadata\.lineage must be an object\./);
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
}

console.log('verify-training-provenance.test: ok');
