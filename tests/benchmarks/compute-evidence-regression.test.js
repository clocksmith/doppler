import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { computeSampleStats } from '../../src/debug/stats.js';
import { KernelBenchmark, formatBenchmarkResult } from '../kernels/harness/benchmark.js';
import { validateSuite, validateSample } from '../../benchmarks/compute/contract.js';
import { analyzeComparison, runEvidenceControls, runStatisticsControls, summarize } from '../../benchmarks/compute/analysis.js';
import { createLinearInputs, referenceLinear, compareOutput, runOracleControls, createLinearPlan } from '../../benchmarks/compute/linear.js';

const suite = JSON.parse(readFileSync(new URL('../../benchmarks/compute/suite.json', import.meta.url), 'utf8'));
const cases = [];
function check(name, run) {
  try {
    run();
    cases.push({ name, passed: true });
  } catch (error) {
    cases.push({ name, passed: false, error: error.stack });
  }
}
function near(actual, expected) {
  assert.ok(Math.abs(actual - expected) < 1e-9, `${actual} differs from ${expected}`);
}

check('shared statistics preserve latency tails when explicitly requested', () => {
  const raw = [...Array(18).fill(1), 20, 30];
  const before = raw.slice();
  const result = computeSampleStats(raw, { outlierPolicy: 'none' });
  near(result.p95, 20.5);
  near(result.p99, 28.1);
  near(result.mean, 3.4);
  assert.equal(result.outliersRemoved, 0);
  assert.equal(result.samplesAfterOutlierRemoval, raw.length);
  assert.deepEqual(raw, before);
});

check('legacy IQR callers retain their existing default behavior', () => {
  const result = computeSampleStats([...Array(18).fill(1), 20, 30]);
  assert.equal(result.outliersRemoved, 2);
  assert.equal(result.p95, 1);
  assert.equal(result.samplesAfterOutlierRemoval, 18);
});

check('unknown outlier policy fails instead of silently selecting a population', () => {
  assert.throws(() => computeSampleStats([1, 2], { outlierPolicy: 'trim' }), /Unknown sample outlier policy/);
});

check('shared statistics retain empty and singleton compatibility', () => {
  assert.equal(computeSampleStats([], { outlierPolicy: 'none' }).samples, 0);
  const single = computeSampleStats([3], { outlierPolicy: 'none' });
  assert.equal(single.p99, 3);
  assert.equal(single.stdDev, 0);
});

check('benchmark summaries reject empty and non-finite populations', () => {
  for (const values of [[], [1, NaN], [1, Infinity]]) assert.throws(() => summarize(values), /missing or invalid samples/);
});

// No GPU is mocked or executed: computeStats only uses CPU observation helpers.
const benchmark = new KernelBenchmark({ features: new Set(['timestamp-query']) });
check('kernel reporting preserves tails and identifies its actual wall clock', () => {
  const result = benchmark.computeStats([...Array(18).fill(1), 20, 30], 'tail-fixture');
  near(result.p95Ms, 20.5);
  near(result.p99Ms, 28.1);
  assert.equal(result.outlierPolicy, 'none');
  assert.equal(result.gpuTimestampsUsed, false);
  assert.equal(result.timingSource, 'performance.now');
  assert.equal(result.timingScope, 'kernelFn-and-queue-completion');
  assert.equal(result.samplesAfterOutlierRemoval, result.samples);
});

check('kernel reporting rejects invalid timing values', () => {
  for (const values of [[NaN], [Infinity], [-1], null]) assert.throws(() => benchmark.computeStats(values, 'invalid'), /finite, non-negative/);
});

check('mean confidence interval is not presented as a median interval', () => {
  const result = benchmark.computeStats([1, 2, 3, 8], 'labels');
  const formatted = formatBenchmarkResult(result);
  const medianLine = formatted.split('\n').find(line => line.includes('Median:'));
  const meanLine = formatted.split('\n').find(line => line.includes('Mean:'));
  assert.ok(!medianLine.includes('CI') && !medianLine.includes('+/-'));
  assert.match(meanLine, /approximate 95% mean CI/);
});

check('default compute suite validates without filling implicit fields', () => {
  const before = structuredClone(suite);
  assert.equal(validateSuite(suite), suite);
  assert.deepEqual(suite, before);
});

check('suite rejects missing and unknown policy fields', () => {
  const missing = structuredClone(suite);
  delete missing.shared.repetitions;
  assert.throws(() => validateSuite(missing), /missing or unknown fields/);
  const extra = structuredClone(suite);
  extra.shared.hiddenTuning = true;
  assert.throws(() => validateSuite(extra), /missing or unknown fields/);
});

check('suite rejects changed non-treatment axes', () => {
  const changed = structuredClone(suite);
  changed.experiments[0].candidate.submission = 'dispatch';
  assert.throws(() => validateSuite(changed), /undeclared axis/);
});

check('suite rejects duplicate shapes and unsupported numerical policy', () => {
  const duplicate = structuredClone(suite);
  duplicate.cases[1].id = duplicate.cases[0].id;
  assert.throws(() => validateSuite(duplicate), /unique path-safe/);
  const numerical = structuredClone(suite);
  numerical.shared.sigmoidClamp = 10;
  assert.throws(() => validateSuite(numerical), /clamp at 15/);
});

check('suite enforces the declared sample and workload envelope', () => {
  const short = structuredClone(suite);
  short.shared.timedPairs = 1;
  assert.throws(() => validateSuite(short), /20\.\.1000/);
  const huge = structuredClone(suite);
  huge.cases[0].m = 4096;
  huge.cases[0].n = 4096;
  huge.cases[0].k = 4096;
  assert.throws(() => validateSuite(huge), /work envelope/);
});

check('input generation is deterministic and case-seed-sensitive', () => {
  const shape = suite.cases[0];
  assert.deepEqual(createLinearInputs(shape, suite.shared, 0), createLinearInputs(shape, suite.shared, 0));
  assert.notDeepEqual(createLinearInputs(shape, suite.shared, 0).x, createLinearInputs(shape, suite.shared, 1).x);
});

check('scalar oracle matches independently hand-expanded matrix arithmetic', () => {
  const shape = { m: 2, n: 3, k: 2 };
  const inputs = { x: new Float32Array([1, 2, 3, 4]), weights: new Float32Array([1, 0, -1, 2, 1, 0]), bias: new Float32Array([0, 1, -1]) };
  const biased = [5, 3, -2, 11, 5, -4];
  const expected = Float32Array.from(biased, value => value / (1 + Math.exp(-value)));
  assert.deepEqual(referenceLinear(shape, inputs, suite.shared), expected);
});

check('oracle controls reject corruption, incomplete writes, infinities, and guard changes', () => {
  const expected = new Float32Array([1, -2, 3]);
  const controls = runOracleControls(expected, suite.shared);
  assert.equal(controls.length, 6);
  assert.equal(controls[0].observed.passed, true);
  assert.ok(controls.slice(1).every(control => !control.observed.passed));
  assert.equal(compareOutput(new Float32Array(), expected, suite.shared).passed, false);
});

check('fusion preserves the semantic contract and removes only declared intermediates', () => {
  const shape = suite.cases[1];
  const experiment = suite.experiments.find(value => value.id === 'fusion');
  const baseline = createLinearPlan(shape, experiment.baseline, suite);
  const candidate = createLinearPlan(shape, experiment.candidate, suite);
  assert.deepEqual(baseline.semanticContract, candidate.semanticContract);
  assert.equal(baseline.steps.length, 3);
  assert.equal(candidate.steps.length, 1);
  assert.equal(baseline.steps[0].source, candidate.steps[0].source);
  assert.equal(baseline.steps[0].constants.FUSE_EPILOGUE, 0);
  assert.equal(candidate.steps[0].constants.FUSE_EPILOGUE, 1);
  assert.equal(candidate.resources.intermediate, undefined);
  assert.equal(candidate.logicalIntermediateBytesPerOperation, 0);
  assert.equal(baseline.outputBytes, candidate.outputBytes);
});

check('submission treatment preserves the complete shader graph', () => {
  const experiment = suite.experiments.find(value => value.id === 'submission');
  const baseline = createLinearPlan(suite.cases[2], experiment.baseline, suite);
  const candidate = createLinearPlan(suite.cases[2], experiment.candidate, suite);
  assert.deepEqual(baseline.steps, candidate.steps);
  assert.deepEqual(baseline.resources, candidate.resources);
  assert.notEqual(baseline.submission, candidate.submission);
});

function sample(plan, wallMs, outputHash) {
  const repetitions = suite.shared.repetitions;
  const dispatches = plan.steps.length * repetitions;
  return { wallMs, encodeMs: 0.1, submitCpuMs: 0.1, waitAfterLastSubmitMs: 0.1, readbackMs: 0.1, resetMs: 0.1,
    correctness: { passed: true }, gpu: null, outputHash,
    counters: { dispatches, submissions: plan.submission === 'batch' ? 1 : dispatches, readbackSubmissions: 1, outputBytesRead: plan.outputBytes,
      workgroups: plan.steps.reduce((total, step) => total + step.dispatch.reduce((a, b) => a * b, 1), 0) * repetitions } };
}

function comparison() {
  const experiment = suite.experiments[0];
  const plans = Object.fromEntries(['baseline', 'candidate'].map(role => [role, createLinearPlan(suite.cases[0], experiment[role], suite)]));
  const outputs = { baseline: { sha256: 'cpu-fixture-baseline' }, candidate: { sha256: 'cpu-fixture-candidate' } };
  const pairs = Array.from({ length: suite.shared.timedPairs }, (_, index) => ({
    index, order: index % 2 ? ['candidate', 'baseline'] : ['baseline', 'candidate'],
    baseline: sample(plans.baseline, 8 + (index % 3) * 0.2, outputs.baseline.sha256),
    candidate: sample(plans.candidate, 6 + (index % 3) * 0.2, outputs.candidate.sha256),
  }));
  return { plans, outputs, pairs };
}

check('paired estimator uses candidate-minus-baseline and family correction', () => {
  const result = analyzeComparison(comparison(), suite, 0);
  near(result.pairedDifference.mean, -0.25);
  near(result.pairedMeanDifferenceInterval.lower, -0.25);
  near(result.pairedMeanDifferenceInterval.upper, -0.25);
  assert.equal(result.pairedMeanDifferenceInterval.familySize, suite.cases.length * suite.experiments.length);
  assert.equal(result.pairedMeanDifferenceInterval.alpha, suite.shared.familyAlpha / 8);
  assert.equal(result.outcome, 'local-improvement');
  assert.equal(result.productionPromotionAllowed, false);
});

check('equal timings cannot become a local improvement', () => {
  const row = comparison();
  for (const pair of row.pairs) pair.candidate.wallMs = pair.baseline.wallMs;
  const result = analyzeComparison(row, suite, 0);
  assert.equal(result.outcome, 'inconclusive');
  assert.equal(result.improvementPercent, 0);
});

check('a slower candidate is reported as a regression', () => {
  const row = comparison();
  for (const pair of row.pairs) pair.candidate.wallMs = pair.baseline.wallMs + 2;
  assert.equal(analyzeComparison(row, suite, 0).outcome, 'candidate-regression');
});

check('negative evidence controls reject all declared invalid sample classes', () => {
  const controls = runEvidenceControls(comparison(), suite, 0);
  assert.equal(controls.length, 7);
  assert.ok(controls.every(control => control.rejected));
});

check('coverage and workgroup drift cannot pass the sample boundary', () => {
  const row = comparison();
  const truncated = structuredClone(row.pairs[0].candidate);
  truncated.counters.outputBytesRead -= 4;
  assert.throws(() => validateSample(truncated, row.plans.candidate, suite.shared.repetitions), /Output coverage/);
  const skipped = structuredClone(row.pairs[0].candidate);
  skipped.counters.workgroups -= 1;
  assert.throws(() => validateSample(skipped, row.plans.candidate, suite.shared.repetitions), /workgroup count/);
});

check('retained statistics control remains executable', () => {
  assert.equal(runStatisticsControls().passed, true);
});

const failures = cases.filter(value => !value.passed);
console.log(JSON.stringify({ schema: 'doppler.compute-regressions/v1', scope: 'CPU contract regression checks; no GPU or performance evidence', passed: failures.length === 0, total: cases.length, failures: failures.length, cases }, null, 2));
if (failures.length) process.exitCode = 1;
