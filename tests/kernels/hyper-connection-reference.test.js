import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import {
  boundRatio,
  fixtureInput,
  fixtureParameter,
  gatedResidualReference,
} from './hyper-connection-reference.js';

// Contract: the float64 gated-residual reference reproduces the pinned upstream
// Qwen4ExpTextGatedResidual outputs (executed on torch CPU, F32) within its
// derived F32 bound, and the bound still separates plausible wrong semantics.
const fixtures = JSON.parse(await readFile(
  new URL('../../artifacts/qwen38-flash-next-20261013/reference-fixtures.json', import.meta.url), 'utf8'));
const residual = fixtures.gatedResidual;
const geometry = { tokens: 2, streams: 4, width: 2560, rank: 320, eps: 1e-6 };
const hc = geometry.streams * geometry.width;
const sha = (array) => createHash('sha256').update(Buffer.from(array.buffer, array.byteOffset, array.byteLength)).digest('hex');

const weights = {
  norm: fixtureParameter(hc, 0),
  down: fixtureParameter(geometry.rank * hc, 1),
  up: fixtureParameter(hc * geometry.rank, 2),
  inject: fixtureParameter(geometry.streams * hc, 3),
};
// The recipes reproduce the upstream parameter and input bytes exactly.
for (const [entry, values] of [[residual.weights[0], weights.norm], [residual.weights[1], weights.down],
  [residual.weights[2], weights.up], [residual.weights[3], weights.inject]]) {
  assert.equal(sha(values), entry.sha256, entry.name);
}
const input = fixtureInput(geometry.tokens * hc);
assert.equal(sha(input), residual.inputSha256);

const reference = gatedResidualReference(input, weights, geometry);
const upstreamMixed = Float64Array.from(residual.mixed.values[0].flat());
const upstreamInjection = Float64Array.from(residual.injection.values[0].flat());
const mixed = boundRatio(upstreamMixed, reference.branch.value, reference.branch.bound);
const injection = boundRatio(upstreamInjection, reference.injection.value, reference.injection.bound);
assert.ok(mixed.ratio <= 1, `upstream branch input outside the derived bound: ${mixed.ratio}`);
assert.ok(injection.ratio <= 1, `upstream injection weights outside the derived bound: ${injection.ratio}`);

// Discrimination: each alternative semantics exceeds twice the bound somewhere.
const variant = (patch) => gatedResidualReference(input, { ...weights, ...patch.weights }, { ...geometry, ...patch.geometry });
const noOffset = variant({ weights: { norm: weights.norm.map((w) => w - 1) } }); // weight instead of 1 + weight
const alternatives = {
  'norm weight without +1': noOffset.branch.value,
  'mean replaced by sum': reference.branch.value.map((value) => value * geometry.streams),
};
for (const [name, values] of Object.entries(alternatives)) {
  assert.ok(boundRatio(values, reference.branch.value, reference.branch.bound).ratio > 2, `${name} is not distinguished`);
}
const injectionAlternatives = {
  'injection without the factor 2': reference.injection.value.map((value) => value / 2),
  'injection without division by streams': reference.injectLogits.value.map((value) => 2 / (1 + Math.exp(-value))),
};
for (const [name, values] of Object.entries(injectionAlternatives)) {
  assert.ok(boundRatio(values, reference.injection.value, reference.injection.bound).ratio > 2, `${name} is not distinguished`);
}

console.log(JSON.stringify({
  test: 'hyper-connection-reference',
  upstreamBranchRatio: mixed.ratio,
  upstreamInjectionRatio: injection.ratio,
  maxBranchBound: Math.max(...reference.branch.bound),
  scope: 'CPU reference vs pinned upstream synthetic fixture; no GPU or learned weights',
}));
