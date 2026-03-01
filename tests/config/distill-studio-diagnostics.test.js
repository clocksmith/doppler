import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

function runNodeScript(args) {
  return new Promise((resolve, reject) => {
    execFile('node', args, { cwd: process.cwd() }, (error, stdout, stderr) => {
      if (error) {
        reject(new Error(`${error.message}\n${stderr || stdout}`));
        return;
      }
      resolve({ stdout, stderr });
    });
  });
}

const tempDir = await mkdtemp(join(tmpdir(), 'doppler-distill-diag-test-'));
try {
  const reportPath = join(tempDir, 'report.json');
  await writeFile(reportPath, JSON.stringify({
    suite: 'training',
    modelId: 'toy-model',
    metrics: null,
  }, null, 2), 'utf8');
  const result = await runNodeScript([
    'tools/distill-studio-diagnostics.mjs',
    '--report',
    reportPath,
  ]);
  assert.match(result.stdout, /"ok": true/);
} finally {
  await rm(tempDir, { recursive: true, force: true });
}

console.log('distill-studio-diagnostics.test: ok');
