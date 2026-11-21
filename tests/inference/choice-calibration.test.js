import assert from 'node:assert/strict';
import test from 'node:test';
import { CHOICE_SCORING_CONTRACT } from '../../src/config/choice-scoring.js';
import {
  CALIBRATED_CHOICE_RESULT_SCHEMA,
  CHOICE_CALIBRATION_SCHEMA,
  applyChoiceTemperatureCalibration,
  computeChoiceProbabilities,
  evaluateChoiceCalibration,
  fitChoiceTemperatureCalibration,
} from '../../src/tooling-exports/structured.js';

function makeResult(logits, ids = ['a', 'b', 'c', 'd']) {
  let bestIdx = 0;
  for (let i = 1; i < logits.length; i += 1) {
    if (logits[i] > logits[bestIdx]) bestIdx = i;
  }
  return {
    schema: CHOICE_SCORING_CONTRACT.resultSchema,
    interpretation: CHOICE_SCORING_CONTRACT.interpretation,
    calibration: CHOICE_SCORING_CONTRACT.calibration,
    promptTokenCount: 12,
    selectedId: ids[bestIdx],
    choices: logits.map((logit, idx) => ({
      id: ids[idx],
      label: ` ${ids[idx].toUpperCase()}`,
      tokenId: 100 + idx,
      logit,
    })),
  };
}

test('temperature scaling calibrates overconfident raw choice logits while preserving argmax selection', () => {
  // Construct 20 examples where raw logits are 4x over-scaled relative to true 70% accuracy.
  const examples = [];
  for (let i = 0; i < 20; i += 1) {
    const correct = i % 10 < 7;
    const rawLogits = [3.6, 0.4, -0.8, -1.2];
    examples.push({
      result: makeResult(rawLogits),
      targetId: correct ? 'a' : 'b',
    });
  }

  const rawMetrics = evaluateChoiceCalibration(examples, { temperature: 1, bins: 5 });
  assert.equal(rawMetrics.accuracy, 0.7);

  const fitted = fitChoiceTemperatureCalibration(examples, { bins: 5, datasetId: 'synthetic-overconfident-20' });
  assert.equal(fitted.schema, CHOICE_CALIBRATION_SCHEMA);
  assert.equal(fitted.method, 'temperature-scaling');
  assert.equal(fitted.datasetId, 'synthetic-overconfident-20');
  assert(fitted.temperature > 1.5, `Expected temperature > 1.5 to soften overconfident logits, got ${fitted.temperature}`);
  assert.equal(fitted.after.accuracy, fitted.before.accuracy);
  assert(fitted.after.meanLogLoss < fitted.before.meanLogLoss);
  assert(fitted.after.brierScore < fitted.before.brierScore);
  assert(fitted.after.ece < fitted.before.ece);

  const sample = makeResult([3.6, 0.4, -0.8, -1.2]);
  const calibrated = applyChoiceTemperatureCalibration(sample, fitted);
  assert.equal(calibrated.schema, CALIBRATED_CHOICE_RESULT_SCHEMA);
  assert.equal(calibrated.selectedId, sample.selectedId);
  const probSum = calibrated.choices.reduce((acc, c) => acc + c.probability, 0);
  assert(Math.abs(probSum - 1) < 1e-12);
  assert(calibrated.choices[0].probability > calibrated.choices[1].probability);
  assert(calibrated.choices[0].probability < computeChoiceProbabilities([3.6, 0.4, -0.8, -1.2], 1)[0]);
});

test('choice calibration rejects malformed inputs explicitly', () => {
  assert.throws(() => computeChoiceProbabilities([1], 1), /at least two finite logits/);
  assert.throws(() => computeChoiceProbabilities([1, NaN], 1), /at least two finite logits/);
  assert.throws(() => computeChoiceProbabilities([1, 2], 0), /positive finite temperature/);
  assert.throws(() => evaluateChoiceCalibration([]), /at least one labeled example/);
  assert.throws(
    () => evaluateChoiceCalibration([{ result: makeResult([1, 2]), targetId: 'missing' }]),
    /absent from result choices/
  );
  assert.throws(() => fitChoiceTemperatureCalibration([{ result: makeResult([1, 2]), targetId: 'a' }], { minTemperature: 2, maxTemperature: 1 }), /0 < minTemperature < maxTemperature/);
});
