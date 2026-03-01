import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
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

const tempDir = await mkdtemp(join(tmpdir(), 'doppler-distill-mvp-test-'));
try {
  const teacherPath = join(tempDir, 'teacher.json');
  const studentPath = join(tempDir, 'student.json');
  const holdoutPath = join(tempDir, 'holdout.json');
  const outPath = join(tempDir, 'out.json');
  await writeFile(teacherPath, JSON.stringify({
    suite: 'bench',
    modelId: 'teacher',
    metrics: {
      trainingMetricsReport: [
        { step: 1, total_loss: 1.0 },
        { step: 2, total_loss: 0.8 },
      ],
    },
  }, null, 2), 'utf8');
  await writeFile(studentPath, JSON.stringify({
    suite: 'bench',
    modelId: 'student',
    metrics: {
      trainingMetricsReport: [
        { step: 1, total_loss: 1.1 },
        { step: 2, total_loss: 0.9 },
      ],
    },
  }, null, 2), 'utf8');
  await writeFile(holdoutPath, JSON.stringify([{ id: 'a' }, { id: 'b' }], null, 2), 'utf8');

  await runNodeScript([
    'tools/distill-studio-mvp.mjs',
    'branch-compare',
    '--teacher',
    teacherPath,
    '--student',
    studentPath,
    '--out',
    outPath,
  ]);
  const branchCompare = JSON.parse(await readFile(outPath, 'utf8'));
  assert.equal(branchCompare.mode, 'branch-compare');
  assert.equal(branchCompare.comparedSteps, 2);

  await runNodeScript([
    'tools/distill-studio-mvp.mjs',
    'mini-eval',
    '--teacher',
    teacherPath,
    '--student',
    studentPath,
    '--holdout',
    holdoutPath,
    '--out',
    outPath,
  ]);
  const miniEval = JSON.parse(await readFile(outPath, 'utf8'));
  assert.equal(miniEval.mode, 'mini-eval');
  assert.equal(miniEval.holdoutSize, 2);
} finally {
  await rm(tempDir, { recursive: true, force: true });
}

console.log('distill-studio-mvp.test: ok');
