import assert from 'node:assert/strict';
import { pairLinearCaptures, recurrentReference } from './recurrent-reference.js';
import { recurrentInterventionOperands, interveneRecurrentShader } from './recurrent-intervention.js';
import { observeRecurrentShader } from './recurrent-observation.js';
import { readFile } from 'node:fs/promises';
import { buildRecurrentAccumulationDiagnostic } from './recurrent-accumulation-diagnostic.js';
import { recurrentReferenceStep } from './recurrent-reference-step.js';

const tensor = (role, values) => {
  const bytes = Buffer.from(Float32Array.from(values).buffer);
  return { role, bytes: bytes.length, data: bytes.toString('base64') };
};
const coordinate = { layerIdx: 0, step: 0, prompt: 0, dispatch: 0 };
const params = { numTokens: 2, numVHeads: 1, numKHeads: 1, headKDim: 1, headVDim: 1,
  qRep: 1, valueDim: 1, qSize: 1, kSize: 1, convDim: 3, normMode: 'shared',
  qkL2NormEps: 0, rmsNormEps: 1, abPacked: false, qkvzPacked: false, bProjOffsetElements: 0 };
const input = { boundary: 'linear-inputs', ...coordinate, params, tensors: [
  tensor('z', [1, 1]), tensor('a', [0, 0]), tensor('b', [0, 0]),
  tensor('dtBias', [0]), tensor('aLog', [0]), tensor('normWeight', [1]), tensor('recurrentState', [2]),
] };
const expected = { boundary: 'linear-outputs', ...coordinate, params,
  tensors: [tensor('convOutput', [1, 1, 3, 1, 1, 5])] };
const original = JSON.stringify({ input, expected });
const result = recurrentReference(input, expected);
const stage = name => {
  const f = result.layout.fields[name]; return [...result.trace.subarray(f.offset, f.offset + f.length)];
};
// Hand-calculated scalar recurrence: decay=1/2, beta=1/2.
// S0=2 -> decayed=1 -> delta=(3-1)/2=1 -> S1=2.
// S1=2 -> decayed=1 -> delta=(5-1)/2=2 -> S2=3.
assert.deepEqual(stage('decayedState'), [1, 1]);
assert.deepEqual(stage('memory'), [1, 1]);
assert.deepEqual(stage('correction'), [1, 2]);
assert.deepEqual(stage('updatedState'), [2, 3]);
assert.deepEqual(stage('rawOutput'), [2, 3]);
assert.deepEqual([...result.finalState], [3]);
assert(Math.abs(result.output[0] - 2 / Math.sqrt(5) / (1 + Math.exp(-1))) < 1e-15);
assert(Math.abs(result.output[1] - 3 / Math.sqrt(10) / (1 + Math.exp(-1))) < 1e-15);
assert.equal(JSON.stringify({ input, expected }), original, 'Reference must not mutate capture inputs');
assert.deepEqual([...recurrentReferenceStep(input, expected, 1, [10]).finalState], [5],
  'Single-step reference must use the supplied state, not the earlier trajectory');
assert.deepEqual([...recurrentReferenceStep(input, expected, 1, [2]).finalState], [3]);
assert.throws(() => recurrentReferenceStep(input, expected, 2, [2]), /assert/i);

const laterInput = { ...input, step: 1 }, laterOutput = { ...expected, step: 1 };
const group = records => ({ captures: [{ records, errors: [] }] });
const pairs = pairLinearCaptures(group([laterInput, expected, input, laterOutput]));
assert.equal(pairs.length, 2);
for (const pair of pairs) assert.equal(pair.input.step, pair.expected.step);
assert.throws(() => pairLinearCaptures(group([input, input, expected])), /Duplicate/);
assert.throws(() => pairLinearCaptures(group([input])), /Unpaired/);
assert.throws(() => pairLinearCaptures(group([{ ...input, step: undefined }, expected])), /coordinates/);
assert.throws(() => pairLinearCaptures(group([input, { ...expected, params: { ...params, numTokens: 1 } }])), /geometry/);
assert.throws(() => recurrentReference({ ...input, tensors: [...input.tensors, input.tensors[0]] }, expected), /exactly one/);
assert.throws(() => recurrentReference({ ...input, params: { ...params, numTokens: 10000000 } }, expected), /diagnostic limit/);

// Sustained nonzero recurrent state, repeated heads, and both packed layouts.
const count = 257;
const p = { ...params, numTokens: count, numVHeads: 2, qRep: 2, valueDim: 2,
  convDim: 4, normMode: 'per_head', qkL2NormEps: 1e-6, abPacked: true,
  bProjOffsetElements: count * 2, qkvzPacked: true };
const z = Array.from({ length: count * 6 }, (_, i) => i % 6 >= 4 ? 1 : 0);
const ab = Array(count * 4).fill(0);
const repeatedInput = { ...input, params: p, tensors: [tensor('z', z), tensor('a', ab), tensor('b', ab),
  tensor('dtBias', [0, 0]), tensor('aLog', [0, 0]), tensor('normWeight', [1, 2]), tensor('recurrentState', [2, 2])] };
const repeatedOutput = { ...expected, params: p, tensors: [tensor('convOutput',
  Array.from({ length: count * 4 }, (_, i) => i % 4 < 2 ? 1 : 3))] };
const sustained = recurrentReference(repeatedInput, repeatedOutput);
assert(sustained.trace.every(Number.isFinite));
assert.equal(sustained.finalState[0], sustained.finalState[1]);
for (let t = 0; t < count; t++) assert.equal(sustained.output[t * 2 + 1], sustained.output[t * 2] * 2);
const shader = await readFile(new URL('../../src/gpu/kernels/gated_delta_recurrent.wgsl', import.meta.url), 'utf8');
for (const intervention of ['normalization', 'gates', 'decay', 'state', 'normalization-gates', 'all']) {
  const packed = recurrentInterventionOperands(result, intervention);
  for (const [name, f] of Object.entries(packed.layout.fields)) {
    assert.deepEqual([...packed.values.subarray(f.offset, f.offset + f.length)], stage(name).map(Math.fround));
  }
  assert(interveneRecurrentShader(shader, packed.layout, intervention).includes('reference_values'));
}
assert.throws(() => interveneRecurrentShader('wrong shader', result.layout, 'normalization'), /Ambiguous/);
assert.throws(() => recurrentInterventionOperands(result, 'unknown'), /Unknown/);
assert.throws(() => observeRecurrentShader(shader, result.layout, ['unknown']), /Unknown/);
assert.throws(() => buildRecurrentAccumulationDiagnostic('wrong shader'), /Missing/);
assert(buildRecurrentAccumulationDiagnostic(shader).includes('memory_error'));
const fusedShader = await readFile(new URL('../../src/gpu/kernels/gated_delta_fused_decode.wgsl', import.meta.url), 'utf8');
for (const [source, fused] of [[shader, false], [fusedShader, true]]) {
  const readout = buildRecurrentAccumulationDiagnostic(source, 'readout', fused);
  assert(!readout.includes('memory_error'), 'Readout candidate must preserve the state update');
  assert(readout.includes('output_error'));
  const memory = buildRecurrentAccumulationDiagnostic(source, 'memory', fused);
  assert(!memory.includes('output_error'), 'Memory candidate must preserve readout');
  assert(memory.includes('memory_error'));
}
// Zero Q/K and saturated sigmoid inputs must remain finite with positive eps.
const zeroInput = { ...input, params: { ...params, qkL2NormEps: 1e-6 },
  tensors: input.tensors.filter(t => t.role !== 'b').concat(tensor('b', [-100, 100])) };
const zeroOutput = { ...expected, params: zeroInput.params,
  tensors: [tensor('convOutput', [0, 0, 3, 0, 0, 5])] };
assert.deepEqual([...recurrentReference(zeroInput, zeroOutput).output], [0, 0]);
console.log('recurrent-reference: scalar equations, state, packed layouts, sustained recurrence and capture pairing passed');
