import { computeSampleStats, percentile } from '../../src/debug/stats.js';
import { requireCondition, validateSample } from './contract.js';
import { seededRandom } from './linear.js';

export function summarize(samples) {
  requireCondition(samples.length > 0 && samples.every(value => Number.isFinite(value)), 'Cannot summarize missing or invalid samples.');
  return computeSampleStats(samples, { outlierPolicy: 'none' });
}

export function runStatisticsControls() {
  const samples = [...Array(18).fill(1), 20, 30];
  const retained = summarize(samples);
  requireCondition(Math.abs(retained.p95 - 20.5) < 1e-10 && Math.abs(retained.p99 - 28.1) < 1e-10 && retained.outliersRemoved === 0, 'Latency tails were silently removed.');
  let rejectedInvalidPolicy = false;
  try { computeSampleStats(samples, { outlierPolicy: 'invalid' }); } catch { rejectedInvalidPolicy = true; }
  requireCondition(rejectedInvalidPolicy, 'Unknown outlier policy was accepted.');
  return { passed: true, samples, retained, legacyIqr: computeSampleStats(samples), rejectedInvalidPolicy,
    preChangeObservation: 'The previous kernel summary reported p95=1 and p99=1 for these same twenty observations.' };
}

export function analyzeComparison(row, suite, index) {
  requireCondition(row.pairs.length === suite.shared.timedPairs, 'Comparison has an incomplete fixed sample budget.');
  const repetitions = suite.shared.repetitions;
  const baseline = [];
  const candidate = [];
  const differences = [];
  for (let i = 0; i < row.pairs.length; i += 1) {
    const pair = row.pairs[i];
    requireCondition(pair.index === i && JSON.stringify(pair.order) === JSON.stringify(i % 2 ? ['candidate', 'baseline'] : ['baseline', 'candidate']), 'Pair order does not match the counterbalanced contract.');
    for (const role of ['baseline', 'candidate']) {
      validateSample(pair[role], row.plans[role], repetitions);
      requireCondition(pair[role].outputHash === row.outputs[role].sha256, 'A timed descendant output differs from the saved, independently checked output.');
    }
    const a = pair.baseline.wallMs / repetitions;
    const b = pair.candidate.wallMs / repetitions;
    baseline.push(a);
    candidate.push(b);
    differences.push(b - a);
  }
  const random = seededRandom(suite.shared.seed + index);
  const bootstraps = [];
  for (let i = 0; i < suite.shared.bootstrapResamples; i += 1) {
    let total = 0;
    for (let j = 0; j < differences.length; j += 1) total += differences[Math.floor(random() * differences.length)];
    bootstraps.push(total / differences.length);
  }
  bootstraps.sort((a, b) => a - b);
  const familySize = suite.cases.length * suite.experiments.length;
  const alpha = suite.shared.familyAlpha / familySize;
  const interval = { lower: percentile(bootstraps, alpha * 50), upper: percentile(bootstraps, (1 - alpha / 2) * 100), alpha, familySize, method: 'fixed-budget percentile bootstrap of paired mean(candidate-baseline); Bonferroni family correction' };
  const a = summarize(baseline);
  const b = summarize(candidate);
  const improvementPercent = (a.median - b.median) / a.median * 100;
  const tailNonRegression = b.p95 <= a.p95;
  let outcome = 'inconclusive';
  if (interval.lower > 0) outcome = 'candidate-regression';
  else if (interval.upper < 0 && improvementPercent >= suite.shared.minimumImprovementPercent) outcome = tailNonRegression ? 'local-improvement' : 'mixed-median-and-tail';
  else if (interval.upper < 0) outcome = 'below-meaningful-effect-threshold';
  return { unit: 'ms/completed-operation', baseline: a, candidate: b, pairedDifference: summarize(differences), pairedMeanDifferenceInterval: interval, improvementPercent, tailNonRegression, outcome,
    evidenceScope: 'controlled-operator-ablation-only', productionPromotionAllowed: false,
    tailCaution: `Empirical tails from ${differences.length} pairs, not a tail-latency service guarantee.` };
}

export function runEvidenceControls(row, suite, index) {
  const controls = [];
  const mutations = {
    'skipped-dispatch': copy => { copy.pairs[0].candidate.counters.dispatches -= 1; },
    'false-correctness': copy => { copy.pairs[0].candidate.correctness.passed = false; },
    'missing-pair': copy => { copy.pairs.pop(); },
    'zero-duration': copy => { copy.pairs[0].candidate.wallMs = 0; },
    'wrong-output-identity': copy => { copy.pairs[0].candidate.outputHash = 'sha256:wrong'; },
    'profiling-in-primary-population': copy => { copy.pairs[0].candidate.gpu = { summedPassMs: 0.001 }; },
    'unbalanced-order': copy => { copy.pairs[0].order.reverse(); },
  };
  for (const [id, mutate] of Object.entries(mutations)) {
    const copy = structuredClone(row);
    mutate(copy);
    let error = null;
    try { analyzeComparison(copy, suite, index); } catch (cause) { error = cause.message; }
    requireCondition(error !== null, `Evidence gate accepted the negative control: ${id}`);
    controls.push({ id, rejected: true, error });
  }
  return controls;
}
