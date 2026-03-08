import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

function runNodeScript(args) {
  return spawnSync(process.execPath, args, {
    cwd: process.cwd(),
    encoding: 'utf8',
  });
}

const tempDir = mkdtempSync(path.join(tmpdir(), 'doppler-distill-quality-gate-'));
try {
  const reportPath = path.join(tempDir, 'report.json');
  const outDir = path.join(tempDir, 'out');
  writeFileSync(reportPath, JSON.stringify({
    reportId: 'rep_test_001',
    modelId: 'toy-model',
    runtimePreset: 'training-distill-ci',
    metrics: {
      trainingSchemaVersion: 1,
      trainingMetricsReport: [
        { step: 1, total_loss: 1.0, seed: 1337, kernel_path: 'auto' },
        { step: 2, total_loss: 0.8, seed: 1337, kernel_path: 'auto' },
      ],
      distillDataset: {
        directionCounts: {
          'en->es': 2,
          'es->en': 2,
        },
      },
      checkpointResumeTimeline: [
        { step: 1, resumed: false },
      ],
    },
  }, null, 2), 'utf8');

  const result = runNodeScript([
    'tools/distill-studio-quality-gate.mjs',
    '--report',
    reportPath,
    '--out-dir',
    outDir,
    '--min-steps',
    '2',
    '--max-total-loss',
    '1.5',
  ]);
  assert.equal(result.status, 0, result.stderr);

  const enGate = JSON.parse(readFileSync(path.join(outDir, 'distill-quality-gate-en.json'), 'utf8'));
  const esGate = JSON.parse(readFileSync(path.join(outDir, 'distill-quality-gate-es.json'), 'utf8'));
  const bundle = JSON.parse(readFileSync(path.join(outDir, 'distill-reproducibility-bundle.json'), 'utf8'));

  assert.equal(enGate.pass, true);
  assert.equal(esGate.pass, true);
  assert.equal(enGate.reportId, 'rep_test_001');
  assert.equal(bundle.reportId, 'rep_test_001');
  // success output must go to stdout, not stderr
  assert.match(result.stdout, /\[distill-quality-gate\] wrote/);
  assert.doesNotMatch(result.stderr, /\[distill-quality-gate\] wrote/);
} finally {
  rmSync(tempDir, { recursive: true, force: true });
}

console.log('distill-studio-quality-gate.test: ok');
