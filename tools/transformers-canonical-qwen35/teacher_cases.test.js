import test from 'node:test';
import assert from 'node:assert/strict';
import { teacherCases } from './teacher_cases.js';

const teacher = { contract: { maxNewTokens: 3, eosTokenIds: [90, 91] }, prompts: [{ id: 'a',
  promptIds: [1, 2], generatedTokenIds: [3, 4, 5], endedByEOS: false,
  snapshots: [{ step: 0, file: 'zero' }, { step: 2, file: 'two' }] }] };

test('teacher case driver feeds actual engine tokens after a divergence', () => {
  const results = [], driver = teacherCases(teacher, results);
  assert.deepEqual(driver.next().value.inputIds, [1, 2]);
  const second = driver.next(7).value;
  assert.deepEqual(second.inputIds, [7]);
  assert.equal(second.expectedTokenId, 4);
  assert.equal(second.teacherPrefixAligned, false);
  const third = driver.next(8).value;
  assert.deepEqual(third.inputIds, [8]);
  assert.equal(third.positionOffset, 3);
  assert.equal(third.logits[0].file, 'two');
  assert.equal(driver.next(9).done, true);
  assert.deepEqual(results[0].generatedTokenIds, [7, 8, 9]);
  assert.equal(results[0].exactTokens, false);
  assert.equal(results[0].completedBudget, true);
});

test('either frozen EOS token ends generation without forced continuation', () => {
  for (const eos of teacher.contract.eosTokenIds) {
    const results = [], driver = teacherCases(teacher, results);
    driver.next();
    assert.equal(driver.next(eos).done, true);
    assert.deepEqual(results[0].generatedTokenIds, [eos]);
    assert.equal(results[0].endedByEOS, true);
    assert.equal(results[0].completedBudget, false);
  }
});
