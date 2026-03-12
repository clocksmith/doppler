import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { closeSync, mkdtempSync, openSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

function runNodeScript(args) {
  const logDir = mkdtempSync(path.join(tmpdir(), 'doppler-distill-mvp-run-'));
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

const tempDir = mkdtempSync(path.join(tmpdir(), 'doppler-distill-mvp-test-'));
try {
  const teacherPath = path.join(tempDir, 'teacher.json');
  const studentPath = path.join(tempDir, 'student.json');
  const holdoutPath = path.join(tempDir, 'holdout.json');
  const invalidHoldoutPath = path.join(tempDir, 'holdout-invalid.json');
  const outPath = path.join(tempDir, 'out.json');
  writeFileSync(teacherPath, JSON.stringify({
    suite: 'bench',
    modelId: 'teacher',
    metrics: {
      trainingMetricsReport: [
        { step: 1, total_loss: 1.0 },
        { step: 2, total_loss: 0.8 },
      ],
    },
  }, null, 2), 'utf8');
  writeFileSync(studentPath, JSON.stringify({
    suite: 'bench',
    modelId: 'student',
    metrics: {
      trainingMetricsReport: [
        { step: 1, total_loss: 1.1 },
        { step: 2, total_loss: 0.9 },
      ],
    },
  }, null, 2), 'utf8');
  writeFileSync(holdoutPath, JSON.stringify([{ id: 'a' }, { id: 'b' }], null, 2), 'utf8');
  writeFileSync(invalidHoldoutPath, JSON.stringify({ id: 'not-array' }, null, 2), 'utf8');

  {
    const result = runNodeScript([
      'tools/distill-studio-mvp.js',
      'replay-teacher',
      '--teacher',
      teacherPath,
      '--out',
      outPath,
    ]);
    assert.equal(result.code, 0);
    const replay = JSON.parse(readFileSync(outPath, 'utf8'));
    assert.equal(replay.mode, 'replay-teacher');
    assert.equal(Array.isArray(replay.timeline), true);
    assert.equal(replay.timeline.length, 2);
  }

  {
    const result = runNodeScript([
      'tools/distill-studio-mvp.js',
      'branch-compare',
      '--teacher',
      teacherPath,
      '--student',
      studentPath,
      '--out',
      outPath,
    ]);
    assert.equal(result.code, 0);
    const branchCompare = JSON.parse(readFileSync(outPath, 'utf8'));
    assert.equal(branchCompare.mode, 'branch-compare');
    assert.equal(branchCompare.comparedSteps, 2);
  }

  {
    const result = runNodeScript([
      'tools/distill-studio-mvp.js',
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
    assert.equal(result.code, 0);
    const miniEval = JSON.parse(readFileSync(outPath, 'utf8'));
    assert.equal(miniEval.mode, 'mini-eval');
    assert.equal(miniEval.holdoutSize, 2);
  }

  {
    const result = runNodeScript([
      'tools/distill-studio-mvp.js',
      'replay-teacher',
    ]);
    assert.equal(result.code, 1);
    assert.match(result.stderr, /--teacher is required\./);
  }

  {
    const result = runNodeScript([
      'tools/distill-studio-mvp.js',
      'branch-compare',
      '--teacher',
      teacherPath,
    ]);
    assert.equal(result.code, 1);
    assert.match(result.stderr, /--student is required for branch-compare\./);
  }

  {
    const result = runNodeScript([
      'tools/distill-studio-mvp.js',
      'mini-eval',
      '--teacher',
      teacherPath,
      '--student',
      studentPath,
      '--holdout',
      invalidHoldoutPath,
    ]);
    assert.equal(result.code, 1);
    assert.match(result.stderr, /--holdout must be a JSON array\./);
  }

  {
    const result = runNodeScript([
      'tools/distill-studio-mvp.js',
      'not-a-mode',
      '--teacher',
      teacherPath,
    ]);
    assert.equal(result.code, 1);
    assert.match(result.stderr, /Unknown mode: not-a-mode/);
  }

  {
    const result = runNodeScript([
      'tools/distill-studio-mvp.js',
      'replay-teacher',
      '--teacher',
      teacherPath,
      '--nope',
      'x',
    ]);
    assert.equal(result.code, 1);
    assert.match(result.stderr, /Unknown flag: --nope/);
  }

  // pulsePass: student within 1.25x teacher loss -> true
  {
    const teacherPassPath = path.join(tempDir, 'teacher-pass.json');
    const studentPassPath = path.join(tempDir, 'student-pass.json');
    const outPassPath = path.join(tempDir, 'out-mini-pass.json');
    writeFileSync(teacherPassPath, JSON.stringify({
      metrics: { trainingMetricsReport: [{ total_loss: 1.0 }] },
    }, null, 2), 'utf8');
    writeFileSync(studentPassPath, JSON.stringify({
      metrics: { trainingMetricsReport: [{ total_loss: 1.2 }] },
    }, null, 2), 'utf8');
    const result = runNodeScript([
      'tools/distill-studio-mvp.js', 'mini-eval',
      '--teacher', teacherPassPath, '--student', studentPassPath,
      '--out', outPassPath,
    ]);
    assert.equal(result.code, 0);
    const out = JSON.parse(readFileSync(outPassPath, 'utf8'));
    assert.equal(out.pulsePass, true, 'student 1.2 <= teacher 1.0 * 1.25 should pass');
  }

  // pulsePass: student exceeds 1.25x teacher loss -> false
  {
    const teacherFailPath = path.join(tempDir, 'teacher-fail.json');
    const studentFailPath = path.join(tempDir, 'student-fail.json');
    const outFailPath = path.join(tempDir, 'out-mini-fail.json');
    writeFileSync(teacherFailPath, JSON.stringify({
      metrics: { trainingMetricsReport: [{ total_loss: 1.0 }] },
    }, null, 2), 'utf8');
    writeFileSync(studentFailPath, JSON.stringify({
      metrics: { trainingMetricsReport: [{ total_loss: 1.3 }] },
    }, null, 2), 'utf8');
    const result = runNodeScript([
      'tools/distill-studio-mvp.js', 'mini-eval',
      '--teacher', teacherFailPath, '--student', studentFailPath,
      '--out', outFailPath,
    ]);
    assert.equal(result.code, 0);
    const out = JSON.parse(readFileSync(outFailPath, 'utf8'));
    assert.equal(out.pulsePass, false, 'student 1.3 > teacher 1.0 * 1.25 should fail');
  }

  // resolveTrainingMetrics fallback: results[] path emits a warning
  {
    const fallbackTeacherPath = path.join(tempDir, 'teacher-fallback.json');
    const fallbackStudentPath = path.join(tempDir, 'student-fallback.json');
    const fallbackOutPath = path.join(tempDir, 'out-fallback.json');
    writeFileSync(fallbackTeacherPath, JSON.stringify({
      results: [{ metrics: { trainingMetricsReport: [{ step: 1, total_loss: 1.0 }] } }],
    }, null, 2), 'utf8');
    writeFileSync(fallbackStudentPath, JSON.stringify({
      results: [{ metrics: { trainingMetricsReport: [{ step: 1, total_loss: 0.9 }] } }],
    }, null, 2), 'utf8');
    const result = runNodeScript([
      'tools/distill-studio-mvp.js', 'branch-compare',
      '--teacher', fallbackTeacherPath, '--student', fallbackStudentPath,
      '--out', fallbackOutPath,
    ]);
    assert.equal(result.code, 0);
    assert.match(result.stderr, /resolveTrainingMetrics: using fallback results\[\] path/);
  }
} finally {
  rmSync(tempDir, { recursive: true, force: true });
}

console.log('distill-studio-mvp.test: ok');
