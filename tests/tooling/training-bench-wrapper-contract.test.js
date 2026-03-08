import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

import {
  parseArgs as parseCompareUlArgs,
} from '../../tools/compare-ul-runs.mjs';
import {
  parseArgs as parseDistillArgs,
  loadWorkloadConfig as loadDistillWorkloadConfig,
  resolveDistillWorkloadOptions,
  resolveResumeOverrideFields,
} from '../../tools/run-distill-bench.mjs';
import {
  parseArgs as parseUlArgs,
} from '../../tools/run-ul-bench.mjs';

assert.throws(
  () => parseCompareUlArgs(['node', 'tools/compare-ul-runs.mjs', '--left']),
  /Missing value for --left/
);

assert.throws(
  () => parseDistillArgs(['node', 'tools/run-distill-bench.mjs', '--surface']),
  /Missing value for --surface/
);

assert.throws(
  () => parseUlArgs(['node', 'tools/run-ul-bench.mjs', '--workload']),
  /Missing value for --workload/
);

{
  const workload = await loadDistillWorkloadConfig('tiny');
  const resolved = resolveDistillWorkloadOptions({
    mode: 'bench',
    trainingBenchSteps: null,
    stageASteps: null,
    stageBSteps: null,
    checkpointEvery: null,
    distillDatasetPath: null,
    distillSourceLangs: null,
    distillTargetLangs: null,
    distillPairAllowlist: null,
    strictPairContract: false,
  }, workload);

  assert.equal(resolved.trainingSchemaVersion, 1);
  assert.equal(resolved.benchSteps, 2);
  assert.equal(resolved.stageASteps, 2);
  assert.equal(resolved.stageBSteps, 2);
  assert.equal(resolved.teacherModelId, 'translategemma-4b-it-wq4k-ef16-hf16');
  assert.equal(resolved.studentModelId, 'gemma-3-1b-it-wq4k-ef16-hf16');
  assert.equal(resolved.distillDatasetId, 'distill-en-es-toy');
  assert.match(resolved.distillDatasetPath, /distill-en-es-toy\.jsonl$/);
  assert.equal(resolved.distillLanguagePair, 'en-es');
  assert.deepEqual(resolved.distillSourceLangs, ['en']);
  assert.deepEqual(resolved.distillTargetLangs, ['es']);
  assert.deepEqual(resolved.distillPairAllowlist, ['en->es']);
  assert.equal(resolved.strictPairContract, true);
  assert.equal(resolved.runtimeConfigJson, null);
}

assert.throws(
  () => resolveResumeOverrideFields({
    forceResume: true,
    forceResumeReason: 'manual override',
    forceResumeSource: null,
    checkpointOperator: null,
  }, {}),
  /forceResume=true requires explicit forceResumeReason and forceResumeSource/
);

{
  const overrides = resolveResumeOverrideFields({
    forceResume: true,
    forceResumeReason: 'manual override',
    forceResumeSource: 'tests',
    checkpointOperator: 'operator-a',
  }, {});
  assert.deepEqual(overrides, {
    forceResume: true,
    forceResumeReason: 'manual override',
    forceResumeSource: 'tests',
    checkpointOperator: 'operator-a',
  });
}

{
  const tempDir = mkdtempSync(path.join(tmpdir(), 'doppler-ul-wrapper-'));
  try {
    const workloadPath = path.join(tempDir, 'bad-ul-workload.json');
    writeFileSync(workloadPath, JSON.stringify({
      schemaVersion: 1,
      id: 'bad-ul',
      description: 'missing stage2',
      seed: 1337,
      trainingSchemaVersion: 1,
      trainingBenchSteps: 2,
      trainingTests: ['ul-stage1'],
    }), 'utf8');

    const result = spawnSync(process.execPath, [
      'tools/run-ul-bench.mjs',
      '--workload',
      workloadPath,
    ], {
      cwd: process.cwd(),
      encoding: 'utf8',
    });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /must include trainingTests: \["ul-stage1", "ul-stage2"\]/);
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
}

console.log('training-bench-wrapper-contract.test: ok');
