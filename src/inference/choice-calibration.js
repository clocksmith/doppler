import { CHOICE_SCORING_CONTRACT } from '../config/choice-scoring.js';

export const CHOICE_CALIBRATION_SCHEMA = 'doppler.choice-calibration/v1';
export const CALIBRATED_CHOICE_RESULT_SCHEMA = 'doppler.calibrated-choice-scoring/v1';

/**
 * @param {ReadonlyArray<number>} logits
 * @param {number} temperature
 * @returns {number[]}
 */
export function computeChoiceProbabilities(logits, temperature = 1) {
  if (!Array.isArray(logits) || logits.length < 2 || logits.some((x) => !Number.isFinite(x))) {
    throw new Error('Choice calibration requires at least two finite logits.');
  }
  if (!Number.isFinite(temperature) || temperature <= 0) {
    throw new Error('Choice calibration requires a positive finite temperature.');
  }
  const invTemp = 1 / temperature;
  let maxScaled = -Infinity;
  for (let i = 0; i < logits.length; i += 1) {
    const scaled = logits[i] * invTemp;
    if (scaled > maxScaled) maxScaled = scaled;
  }
  let sumExp = 0;
  /** @type {number[]} */
  const exps = new Array(logits.length);
  for (let i = 0; i < logits.length; i += 1) {
    const val = Math.exp(logits[i] * invTemp - maxScaled);
    exps[i] = val;
    sumExp += val;
  }
  return exps.map((val) => val / sumExp);
}

/**
 * Evaluate empirical calibration metrics (NLL, Brier score, accuracy, and ECE)
 * over labeled choice-scoring examples at a given temperature.
 * @param {ReadonlyArray<import('./choice-calibration.js').LabeledChoiceScoringExample>} examples
 * @param {{ temperature?: number; bins?: number }} [options]
 * @returns {import('./choice-calibration.js').ChoiceCalibrationMetrics}
 */
export function evaluateChoiceCalibration(examples, options = {}) {
  if (!Array.isArray(examples) || examples.length === 0) {
    throw new Error('Choice calibration evaluation requires at least one labeled example.');
  }
  const temperature = options.temperature ?? 1;
  const bins = options.bins ?? 10;
  if (!Number.isSafeInteger(bins) || bins < 2 || bins > 100) {
    throw new Error('Choice calibration bins must be an integer between 2 and 100.');
  }
  let nllSum = 0;
  let brierSum = 0;
  let correctCount = 0;
  /** @type {Array<{ count: number; confSum: number; accSum: number }>} */
  const bucketStats = Array.from({ length: bins }, () => ({ count: 0, confSum: 0, accSum: 0 }));

  for (const item of examples) {
    const choices = item?.result?.choices;
    if (!Array.isArray(choices) || choices.length < 2) {
      throw new Error('Each calibration example requires a valid choice-scoring result.');
    }
    const targetIdx = choices.findIndex((c) => c.id === item.targetId);
    if (targetIdx < 0) {
      throw new Error(`Calibration targetId "${item.targetId}" is absent from result choices.`);
    }
    const logits = choices.map((c) => c.logit);
    const probs = computeChoiceProbabilities(logits, temperature);
    let bestIdx = 0;
    for (let i = 1; i < probs.length; i += 1) {
      if (probs[i] > probs[bestIdx]) bestIdx = i;
    }
    const correct = bestIdx === targetIdx ? 1 : 0;
    correctCount += correct;
    const pTarget = Math.max(1e-15, probs[targetIdx]);
    nllSum += -Math.log(pTarget);
    let exampleBrier = 0;
    for (let i = 0; i < probs.length; i += 1) {
      const y = i === targetIdx ? 1 : 0;
      const diff = probs[i] - y;
      exampleBrier += diff * diff;
    }
    brierSum += exampleBrier;
    const maxProb = probs[bestIdx];
    const binIdx = Math.min(bins - 1, Math.floor(maxProb * bins));
    bucketStats[binIdx].count += 1;
    bucketStats[binIdx].confSum += maxProb;
    bucketStats[binIdx].accSum += correct;
  }

  const n = examples.length;
  let ece = 0;
  for (const b of bucketStats) {
    if (b.count > 0) {
      ece += (b.count / n) * Math.abs(b.accSum / b.count - b.confSum / b.count);
    }
  }
  return Object.freeze({
    sampleCount: n,
    temperature,
    accuracy: correctCount / n,
    meanLogLoss: nllSum / n,
    brierScore: brierSum / n,
    ece,
  });
}

/**
 * Fit an empirical positive scalar temperature minimizing negative log-likelihood
 * over held-out labeled choice-scoring examples via bounded golden-section search
 * in log-temperature space. Positive temperature scaling preserves strict argmax order.
 * @param {ReadonlyArray<import('./choice-calibration.js').LabeledChoiceScoringExample>} examples
 * @param {{ minTemperature?: number; maxTemperature?: number; bins?: number; datasetId?: string }} [options]
 * @returns {import('./choice-calibration.js').ChoiceTemperatureCalibration}
 */
export function fitChoiceTemperatureCalibration(examples, options = {}) {
  const minT = options.minTemperature ?? 0.05;
  const maxT = options.maxTemperature ?? 20;
  const bins = options.bins ?? 10;
  if (!Number.isFinite(minT) || !Number.isFinite(maxT) || minT <= 0 || maxT <= minT) {
    throw new Error('Choice temperature bounds must satisfy 0 < minTemperature < maxTemperature.');
  }
  const uncalibrated = evaluateChoiceCalibration(examples, { temperature: 1, bins });
  const phi = (Math.sqrt(5) - 1) / 2;
  let low = Math.log(minT);
  let high = Math.log(maxT);
  let c = high - phi * (high - low);
  let d = low + phi * (high - low);
  let fc = evaluateChoiceCalibration(examples, { temperature: Math.exp(c), bins }).meanLogLoss;
  let fd = evaluateChoiceCalibration(examples, { temperature: Math.exp(d), bins }).meanLogLoss;

  for (let iter = 0; iter < 48; iter += 1) {
    if (fc < fd) {
      high = d;
      d = c;
      fd = fc;
      c = high - phi * (high - low);
      fc = evaluateChoiceCalibration(examples, { temperature: Math.exp(c), bins }).meanLogLoss;
    } else {
      low = c;
      c = d;
      fc = fd;
      d = low + phi * (high - low);
      fd = evaluateChoiceCalibration(examples, { temperature: Math.exp(d), bins }).meanLogLoss;
    }
  }
  const fittedTemp = Math.exp((low + high) / 2);
  const calibrated = evaluateChoiceCalibration(examples, { temperature: fittedTemp, bins });
  const bestTemp = calibrated.meanLogLoss <= uncalibrated.meanLogLoss ? fittedTemp : 1;
  const finalMetrics = bestTemp === 1 ? uncalibrated : calibrated;

  return Object.freeze({
    schema: CHOICE_CALIBRATION_SCHEMA,
    method: 'temperature-scaling',
    datasetId: typeof options.datasetId === 'string' && options.datasetId.trim() ? options.datasetId.trim() : null,
    temperature: bestTemp,
    before: uncalibrated,
    after: finalMetrics,
  });
}

/**
 * Apply an explicit empirical temperature calibration to a raw choice-scoring result,
 * preserving raw logits and selectedId while attaching calibrated probabilities.
 * @param {import('../config/choice-scoring.js').ChoiceScoringResult} result
 * @param {Pick<import('./choice-calibration.js').ChoiceTemperatureCalibration, 'schema' | 'method' | 'temperature' | 'datasetId'>} calibration
 * @returns {import('./choice-calibration.js').CalibratedChoiceScoringResult}
 */
export function applyChoiceTemperatureCalibration(result, calibration) {
  if (!result || result.schema !== CHOICE_SCORING_CONTRACT.resultSchema || !Array.isArray(result.choices)) {
    throw new Error('applyChoiceTemperatureCalibration requires a valid ChoiceScoringResult.');
  }
  if (!calibration || calibration.schema !== CHOICE_CALIBRATION_SCHEMA || calibration.method !== 'temperature-scaling') {
    throw new Error('applyChoiceTemperatureCalibration requires a valid ChoiceTemperatureCalibration artifact.');
  }
  const probs = computeChoiceProbabilities(result.choices.map((c) => c.logit), calibration.temperature);
  const choices = result.choices.map((choice, idx) => Object.freeze({
    ...choice,
    probability: probs[idx],
  }));
  return Object.freeze({
    schema: CALIBRATED_CHOICE_RESULT_SCHEMA,
    sourceSchema: result.schema,
    calibration: Object.freeze({
      schema: calibration.schema,
      method: calibration.method,
      temperature: calibration.temperature,
      datasetId: calibration.datasetId ?? null,
    }),
    selectedId: result.selectedId,
    promptTokenCount: result.promptTokenCount,
    choices: Object.freeze(choices),
  });
}
