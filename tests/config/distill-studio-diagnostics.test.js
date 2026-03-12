import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { closeSync, mkdtempSync, openSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

function runNodeScript(args) {
  const logDir = mkdtempSync(path.join(tmpdir(), 'doppler-distill-diag-run-'));
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
  const result = runNodeScript(['tools/distill-studio-diagnostics.js']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /Usage: node tools\/distill-studio-diagnostics\.js --report/);
}

{
  const result = runNodeScript(['tools/distill-studio-diagnostics.js', '--unknown']);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /Unknown flag: --unknown/);
}

const tempDir = mkdtempSync(path.join(tmpdir(), 'doppler-distill-diag-test-'));
try {
  const reportPath = path.join(tempDir, 'report.json');
  writeFileSync(reportPath, JSON.stringify({
    suite: 'training',
    modelId: 'toy-model',
    metrics: null,
  }, null, 2), 'utf8');
  const result = runNodeScript([
    'tools/distill-studio-diagnostics.js',
    '--report',
    reportPath,
  ]);
  assert.equal(result.code, 0);
  assert.match(result.stdout, /"ok": true/);
} finally {
  rmSync(tempDir, { recursive: true, force: true });
}

console.log('distill-studio-diagnostics.test: ok');
