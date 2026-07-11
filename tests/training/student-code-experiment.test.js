import assert from 'node:assert/strict';

import {
  buildGoldStudentCandidate,
  buildOuroborosFailureSignals,
  buildStudentPromotionReport,
  buildStudentReplaySummary,
  buildStudentTrainingDatasets,
  evaluateStudentCandidate,
  loadStudentCodeExperimentContracts,
  parseStudentCandidateOutput,
  renderStudentTaskPrompt,
  selectStudentHoldoutTasks,
  verifyStudentCodeExperimentContracts,
} from '../../tools/lib/student-code-experiment.js';

const report = await verifyStudentCodeExperimentContracts();
assert.equal(report.ok, true, report.errors.join('\n'));
assert.deepEqual(report.labelTasks, { javascript: 6, wgsl: 6 });
assert.equal(report.holdoutTasks, 4);

const contracts = await loadStudentCodeExperimentContracts();
for (const harnessFile of [
  'src/config/kernels/kernel-ref-digests.js',
  'src/experimental/training/optimizer.js',
  'src/gpu/kernels/backward/matmul_backward.wgsl',
  'src/gpu/kernels/backward/utils.js',
  'src/memory/buffer-pool.js',
]) {
  assert.equal(
    contracts.policy.harnessFiles.includes(harnessFile),
    true,
    `student experiment must pin ${harnessFile}`
  );
}
const trainingDatasets = await buildStudentTrainingDatasets({
  contracts,
  teacherRunRoot: 'reports/training/teacher-qualification/doppler-js-wgsl-2026-07-11-v4',
});
assert.deepEqual(trainingDatasets.eligibleAcceptedLaneCounts, { javascript: 6, wgsl: 4 });
assert.equal(trainingDatasets.acceptedLabelCount, 10);
assert.equal(trainingDatasets.datasets.javascript.sourceRowCount, 6);
assert.equal(trainingDatasets.datasets.javascript.materializedRowCount, 12);
assert.equal(trainingDatasets.datasets.wgsl.sourceRowCount, 4);
assert.equal(trainingDatasets.datasets.wgsl.materializedRowCount, 8);
assert.deepEqual(trainingDatasets.datasets.mixed.laneCounts, { javascript: 4, wgsl: 4 });
assert.equal(trainingDatasets.datasets.mixed.materializedRowCount, 16);
const task = selectStudentHoldoutTasks(contracts, ['javascript'])[0];
const prompt = await renderStudentTaskPrompt(contracts, task);
for (const mutation of task.mutations) {
  assert.equal(prompt.includes(mutation.replace), true);
  assert.equal(prompt.includes(mutation.find), false);
}

const gold = buildGoldStudentCandidate(task);
const parsedGold = parseStudentCandidateOutput(JSON.stringify(gold));
assert.equal(parsedGold.schemaValid, true);
assert.deepEqual(parsedGold.candidate, gold);
assert.deepEqual(parsedGold.violations, []);

const wrapped = parseStudentCandidateOutput(`\`\`\`json\n${JSON.stringify(gold)}\n\`\`\``);
assert.equal(wrapped.schemaValid, true);
assert.deepEqual(wrapped.violations.map((violation) => violation.code), ['output_wrapper']);

const constructive = await evaluateStudentCandidate({
  contracts,
  task,
  rawOutput: JSON.stringify(gold),
  variant: 'baseline',
  repetition: 1,
  prompt,
});
assert.equal(constructive.passed, true);
assert.equal(constructive.checks.patchApplicable, true);
assert.equal(constructive.checks.exactSourceRecovery, true);
assert.equal(constructive.checks.validationPassed, true);
assert.equal(constructive.policyViolationCount, 0);

function syntheticRow(variant, lane, repetition, passed) {
  return {
    variant,
    repetition,
    task: { id: `${lane}-holdout`, lane, split: 'student_holdout' },
    outputHash: `${variant}-${lane}-stable`,
    passed,
    checks: {
      patchApplicable: passed,
      exactSourceRecovery: passed,
      validationPassed: passed,
    },
    policyViolationCount: passed ? 0 : 1,
    policyViolations: passed ? [] : [{ code: 'invalid_json', detail: 'observed' }],
    applyErrors: [],
    performance: {
      generationDurationMs: 10,
      completionTokens: 2,
    },
  };
}

const rows = [];
for (let repetition = 1; repetition <= 3; repetition += 1) {
  rows.push(syntheticRow('baseline', 'javascript', repetition, false));
  rows.push(syntheticRow('baseline', 'wgsl', repetition, false));
  rows.push(syntheticRow('javascript', 'javascript', repetition, true));
  rows.push(syntheticRow('wgsl', 'wgsl', repetition, true));
}
const summaries = {
  baseline: buildStudentReplaySummary('baseline', rows),
  javascript: buildStudentReplaySummary('javascript', rows),
  wgsl: buildStudentReplaySummary('wgsl', rows),
};
const promotion = buildStudentPromotionReport(contracts.policy, summaries);
assert.equal(promotion.controlProven, true);
assert.equal(promotion.candidates.specialized.eligible, true);
assert.equal(promotion.candidates.mixed.eligible, false);
assert.equal(
  promotion.challengers.every((challenger) => challenger.status === 'eligible_for_external_trainer'),
  true
);

const failureSignals = buildOuroborosFailureSignals(rows, contracts.policy.policyId);
assert.equal(failureSignals.length, 2);
for (const signal of failureSignals) {
  assert.equal('taskId' in signal, false);
  assert.equal('path' in signal, false);
  assert.equal('prompt' in signal, false);
  assert.equal('output' in signal, false);
  assert.equal('completion' in signal, false);
}

console.log('student-code-experiment.test: ok');
