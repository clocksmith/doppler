import assert from 'node:assert/strict';
import { test } from 'node:test';
import { compileProgram } from '../compiler.js';
import { evaluatePlaintext, loadTriageModel } from '../triage-model.js';
import { TASK_DIGEST, taskCases, verifyTaskDigest } from '../triage-task.js';

test('pinned triage task holds both orders of a word pair out of training', () => {
  assert.equal(verifyTaskDigest(), TASK_DIGEST);
  const key = row => [...row.tokens].sort((a, b) => a - b).join();
  const trainingPairs = new Set(taskCases('train').map(key));
  assert.ok(taskCases('holdout').every(row => !trainingPairs.has(key(row))));
  const { model } = loadTriageModel();
  assert.ok(evaluatePlaintext(model, 'holdout').passed);
});

test('the compiled decode reuses encrypted state and fits its pinned modulus chain', () => {
  const { program } = compileProgram({ profile: 'research-triage-n14-u32', prompt: 2, decodeThrough: 2, triage: true });
  const [prefill, decode] = program.steps;
  assert.equal(prefill.outputLevel, 1);
  assert.equal(decode.outputLevel, 1);
  assert.equal(decode.inputs.length, 1);
  assert.deepEqual(decode.stateIn, prefill.stateOut);
  assert.ok(decode.stateIn.every(({ reg }) => program.regs[reg].level === 5));
  assert.deepEqual(program.rotations, [1, 8, 16]);
  assert.equal(program.galois.length, program.rotations.length);
});
