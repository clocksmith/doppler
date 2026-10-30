import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { compareState } from './compare_state.js';

const oracle = JSON.parse(fs.readFileSync(new URL('./state-threshold-oracle.json', import.meta.url)));
const policy = JSON.parse(fs.readFileSync(new URL('./contract.json', import.meta.url))).textExport;

test('browser state gate matches independent NumPy F32 threshold-boundary oracle', () => {
  assert.equal(policy.stateAbsoluteTolerance, oracle.absoluteTolerance);
  assert.equal(policy.stateRelativeTolerance, oracle.relativeTolerance);
  const result = compareState(Float32Array.from(oracle.actual), Float32Array.from(oracle.reference), policy);
  assert.equal(result.failedElements, oracle.failedElements);
  for (let index = 0; index < oracle.actual.length; index++) {
    const one = compareState(Float32Array.of(oracle.actual[index]), Float32Array.of(oracle.reference[index]), policy);
    assert.equal(one.pass, oracle.elementPass[index], 'F32 boundary element ' + index);
  }
});

test('browser observer rejects nonfinite values and physical dtype substitution', () => {
  assert.equal(compareState(Float32Array.of(NaN, Infinity), Float32Array.of(NaN, 1), policy).pass, false);
  assert.throws(() => compareState(new Uint16Array(2), new Float32Array(2), policy), /physical F32/);
  assert.throws(() => compareState(Float32Array.of(1), Float32Array.of(1), {}), /explicit finite tolerances/);
});
