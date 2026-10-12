// CPU-only analytic boundary witness for a rejected arithmetic experiment.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const shader = await readFile(new URL('../candidate-frequency-residual-rope.wgsl', import.meta.url));
const candidateSha256 = digest(shader);
assert.equal(candidateSha256, '217fcc8c7d5018d27f18ab529bf648ee68ed4d55857eb1bb8d57be802c948dc5');

// Nominal exact-arithmetic reduction intervals, not GPU error bounds.
const s = (Math.SQRT2 - 1) / (Math.SQRT2 + 1);
const r = Math.LN2 / 2;
let factorial15 = 1;
for (let i = 2; i <= 15; i++) factorial15 *= i;
const logTailBound = 2 * s ** 29 / (29 * (1 - s * s));
const expTailBound = Math.exp(r) * r ** 15 / factorial15;

// Legal theta and geometry, below the candidate's enabled position limit.
// Use ideal host-double trig at high so the witness isolates the neglected
// quadratic term rather than shader polynomial/backend rounding defects.
const parameters = { theta: 2, baseDimension: 256, dimension: 125, position: 16382 };
const frequency = parameters.theta ** (-2 * parameters.dimension / parameters.baseDimension);
const angle = parameters.position * frequency;
const high = Math.fround(angle);
const low = angle - high;
const firstOrderCos = Math.cos(high) - Math.sin(high) * low;
const targetCos = Math.cos(angle);
const witness = {
  parameters, frequency, angle, high, low, firstOrderCos, targetCos,
  absoluteError: Math.abs(firstOrderCos - targetCos),
  firstOrderF32: Math.fround(firstOrderCos), targetF32: Math.fround(targetCos),
  leadingOmittedTermMagnitude: Math.abs(Math.cos(high) * low * low / 2),
};
assert(witness.absoluteError > 1e-7);
assert.notEqual(witness.firstOrderF32, witness.targetF32);

console.log(JSON.stringify({
  candidateSha256, checkerSha256: digest(await readFile(new URL(import.meta.url))),
  node: process.version, logTailBound, expTailBound, witness,
  conclusion: 'The neglected phase quadratic can change F32 rounding inside the enabled domain.',
  limits: 'Host binary64 analytic witness; no shader execution, F64 authority change, complete operator reference, or whole-model qualification.',
}, null, 2));
