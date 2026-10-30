import { residentSchedule } from './resident-pairs.js';

function generator(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D2B79F5) | 0;
    let value = Math.imul(state ^ state >>> 15, 1 | state);
    value ^= value + Math.imul(value ^ value >>> 7, 61 | value);
    return ((value ^ value >>> 14) >>> 0) / 4294967296;
  };
}
const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
const ratio = values => Math.exp(mean(values));
function quantile(sorted, probability) {
  const index = (sorted.length - 1) * probability, floor = Math.floor(index);
  return sorted[floor] + (sorted[Math.ceil(index)] - sorted[floor]) * (index - floor);
}

export function validateResidentStatisticsMethod(method) {
  const bootstrap = method.bootstrap;
  if (method.estimator !== 'exp-mean-log-paired-doppler-over-pooled-complete-ms'
      || method.perPromptEstimator !== 'exp-mean-log-paired-ratio'
      || bootstrap.strata !== 'fixed-prompt-id' || bootstrap.resamplingUnit !== 'complete-pair'
      || bootstrap.replacement !== true || bootstrap.random !== 'mulberry32'
      || bootstrap.interval !== 'two-sided-percentile' || bootstrap.quantile !== 'linear-index-n-minus-one'
      || bootstrap.confidence !== 0.95 || !Number.isSafeInteger(bootstrap.seed)
      || bootstrap.seed < 0 || bootstrap.seed > 0xFFFFFFFF
      || !Number.isSafeInteger(bootstrap.replicates) || bootstrap.replicates < 10000) throw new Error('Unsupported frozen estimator');
}

export function summarizeResidentPairs(attempts, prompts, contract, method) {
  validateResidentStatisticsMethod(method);
  const bootstrap = method.bootstrap;
  const schedule = residentSchedule(method, prompts);
  if (attempts.length !== schedule.length || attempts.some((row, index) =>
    Object.entries(schedule[index]).some(([key, value]) => row[key] !== value))) throw new Error('Incomplete or reordered request schedule');
  const measured = attempts.filter(attempt => attempt.warmup === false);
  if (attempts.some(attempt => attempt.status !== 'passed') || measured.length !== method.totalPairs * 2) {
    throw new Error('Incomplete or failed requests cannot produce a performance result');
  }
  const strata = prompts.map(prompt => {
    const requests = measured.filter(attempt => attempt.promptId === prompt.id);
    const pairIds = [...new Set(requests.map(attempt => attempt.pair))];
    if (pairIds.length !== method.pairsPerPrompt || requests.length !== pairIds.length * 2) throw new Error('Incomplete prompt stratum');
    const logs = [], ttftLogs = [];
    for (const pair of pairIds) {
      const rows = requests.filter(attempt => attempt.pair === pair);
      const pooled = rows.find(row => row.engine === 'pooled'), doppler = rows.find(row => row.engine === 'doppler');
      if (rows.length !== 2 || !pooled || !doppler) throw new Error('Incomplete or duplicated pair');
      for (const row of rows) if (![row.completeMs, row.firstTokenMs].every(value => Number.isFinite(value) && value > 0)) {
        throw new Error('Nonpositive paired timing');
      }
      logs.push(Math.log(doppler.completeMs / pooled.completeMs));
      ttftLogs.push(Math.log(doppler.firstTokenMs / pooled.firstTokenMs));
    }
    return { promptId: prompt.id, logs, completeRatio: ratio(logs), firstTokenRatio: ratio(ttftLogs) };
  });
  const random = generator(bootstrap.seed), estimates = [];
  for (let replicate = 0; replicate < bootstrap.replicates; replicate++) {
    const selected = strata.flatMap(stratum => stratum.logs.map(() => stratum.logs[Math.floor(random() * stratum.logs.length)]));
    estimates.push(ratio(selected));
  }
  estimates.sort((a, b) => a - b);
  const completeRatio = ratio(strata.flatMap(stratum => stratum.logs));
  const interval95 = [quantile(estimates, 0.025), quantile(estimates, 0.975)];
  const gates = contract.gates;
  const checks = {
    completeGain: completeRatio <= 1 - gates.minimumCompleteRequestGain,
    upper95: interval95[1] < gates.upper95RatioBelow,
    perPromptComplete: strata.every(stratum => stratum.completeRatio <= 1 + gates.perPromptMaximumCompleteRegression),
    perPromptFirstToken: strata.every(stratum => stratum.firstTokenRatio <= 1 + gates.perPromptMaximumFirstTokenRegression),
  };
  return { completeRatio, interval95, pairs: measured.length / 2, outliersRemoved: 0,
    prompts: strata.map(({ logs, ...stratum }) => stratum), checks, performancePass: Object.values(checks).every(Boolean) };
}
