import assert from 'node:assert/strict';
import { assessRelevance, calibrateThreshold, decisionMetrics } from '../../tools/evaluate-search-components.js';

const row = (answerable, rankingCorrect, score) => ({ answerable, rankingCorrect,
  results: score == null ? [] : [{ score }] });
const rows = [row(true, true, 4), row(true, true, 4), row(false, null, 1)];
const contract = { minimumUsefulAcceptance: 0.9, maximumFalseAcceptance: 0 };
const policy = calibrateThreshold(rows, contract);
assert.equal(policy.threshold, 4);
assert.equal(policy.calibrationPassed, true);
assert.equal(decisionMetrics(rows, 5).usefulAcceptance, 0);
assert.equal(calibrateThreshold([row(true, true, 1), row(false, null, 4)], contract).calibrationPassed, false);
assert.equal(decisionMetrics([row(true, false, 5), row(false, null, 0)], 4).usefulAcceptance, 0);
assert.throws(() => calibrateThreshold([row(true, true, 1)], contract), /Both/);
assert.throws(() => calibrateThreshold([row(true, true, NaN), row(false, null, 0)], contract));
assert.deepEqual(assessRelevance(10, null, 'a'), { outcome: 'unassessed' });
assert.deepEqual(assessRelevance(10, { status: 'candidate' }, 'a'), { outcome: 'unassessed' });
const qualified = { ...policy, rule: 'highest-score-at-least', status: 'qualified', binding: 'pair-a' };
assert.equal(assessRelevance(4, qualified, 'pair-a').outcome, 'match');
assert.equal(assessRelevance(3, qualified, 'pair-a').message, 'No sufficiently relevant result found');
assert.equal(assessRelevance(null, qualified, 'pair-a').outcome, 'abstain');
assert.throws(() => assessRelevance(4, qualified, 'pair-b'), /configuration/);
assert.throws(() => assessRelevance(Infinity, qualified, 'pair-a'), /Non-finite/);
