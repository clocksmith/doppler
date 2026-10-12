import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createSessionController } from '../../src/client/runtime/session-controller.js';
import { createBundledIncrementalDecoder } from '../../src/inference/tokenizers/bundled/incremental-decoder.js';
import { sampleCanonicalGenerationStep } from '../../src/inference/generation-step.js';

const base = { ...JSON.parse(readFileSync(new URL('../fixtures/canonical-generation-policy.json', import.meta.url))),
  maxGeneratedTokens: 8, eosTokenIds: [], stopTokenIds: [], toolTokenIds: [], stopSequences: [] };
const executionIdentity = `sha256:${'e'.repeat(64)}`;
const plan = { phases: { prefill: [], decode: [] }, memoryLayout: {} };
const request = policy => ({ schema: 'doppler.canonical-generation-request/v1', promptTokens: [1, 2], policy, executionIdentity });
const force = id => { const row = new Float32Array(100).fill(-Infinity); row[id] = 0; return row; };
function fixture(scores = () => new Float32Array(100)) {
  const calls = [], released = [];
  const decoder = () => {
    const value = createBundledIncrementalDecoder({ tokenForId: id => ['helloS', 'TOPafter'][id] ?? 'x',
      byteLevel: false, byteDecoder: new Map(), wordPiece: false, splitEveryCharacter: false, wordPiecePrefix: '##' });
    value.clone = () => { throw new Error('Ordinary generation must be scratch-free.'); };
    return value;
  };
  const program = { getGenerationGeometry: () => ({ vocabSize: 100, maxSeqLen: 64 }),
    createRawIncrementalDecoder: decoder, createIncrementalDecoder: decoder,
    reset() { calls.push('reset'); }, releaseStepResult(result) { if (result) released.push(result); },
    close() { calls.push('close'); } };
  let index = 0;
  const controller = createSessionController({ async executePhase(phase, commands, value) {
    calls.push({ phase, ...structuredClone(value.context) });
    return { results: [{ logits: scores(index++) }] };
  } }, { bindSlots(layout, dimensions) { calls.push({ dimensions }); },
    writeSlot() { calls.push('write'); }, assertDeviceAvailable() {},
    releaseTransient() { calls.push('release'); }, releaseAll() { calls.push('releaseAll'); } }, program);
  return { controller, calls, released, stream: p => controller.generateCanonicalTransitions(plan, request(p)) };
}
for (const policy of [{ ...base, maxGeneratedTokens: 0 }, { ...base, totalAcceptedTokenLimit: 2 }]) {
  const run = fixture();
  const events = await Array.fromAsync(run.stream(policy));
  assert.equal(events.length, 1);
  assert.equal(events[0].schema, 'doppler.admission-only/v1');
  assert(!('successor' in events[0]));
  assert.deepEqual(run.calls, [], 'empty admission does not reset, allocate, dispatch or decode');
}
for (const [capacity, limit, expectedCount] of [[2, 64, 1], [3, 64, 2], [64, 3, 1], [64, 5, 3], [64, 65, 8]]) {
  const run = fixture(), policy = { ...base, materializedInputCapacity: capacity, totalAcceptedTokenLimit: limit };
  const events = await Array.fromAsync(run.stream(policy));
  assert.equal(events.length, expectedCount);
  assert.equal(events.at(-1).successor.stopReason, expectedCount === 8 ? 'max-tokens' : 'context-limit');
  const phases = run.calls.filter(item => item.phase);
  assert.equal(phases.length, expectedCount);
  assert.equal(phases[0].phase, 'prefill');
  for (let j = 1; j < phases.length; j++) {
    assert.equal(phases[j].contextTokens.length, j + 2, 'consume only predecessors');
    assert.equal(events[j].successor.materializedTokenCount, j + 2);
    assert.equal(events[j].successor.nextTargetTokenIndex, j + 1);
  }
  assert.equal(run.released.length, phases.length);
}
{
  const run = fixture(), a = run.stream(base), other = fixture(), b = other.stream(base);
  const history = [1, 2];
  for (let j = 0; j < 8; j++) {
    const expected = sampleCanonicalGenerationStep(new Float32Array(100), history, base, j);
    assert.equal((await b.next()).value.acceptedTokenIds[0], expected);
    const result = (await a.next()).value;
    assert.equal(result.acceptedTokenIds[0], expected);
    history.push(expected);
    assert.equal(run.calls.filter(item => item.phase).length, j + 1, 'no model work beyond the caller barrier');
  }
  assert((await a.next()).done); assert((await b.next()).done);
}
{
  const run = fixture(index => force(index)), events = await Array.fromAsync(run.stream({ ...base, stopSequences: ['STOP'] }));
  assert.equal(events.map(item => item.delta).join(''), 'hello');
  assert.equal(events.at(-1).successor.stopReason, 'stop-sequence');
  assert.equal(events.length, 2);
}
{
  const run = fixture();
  const signal = new AbortController();
  const stream = run.controller.generateCanonicalTransitions(plan, { ...request(base), signal: signal.signal });
  await stream.next();
  await assert.rejects(run.stream(base).next(), /already active/);
  await assert.rejects(run.controller.close(), /active generation/);
  signal.abort(); await assert.rejects(stream.next(), /abort/i);
  assert.equal(run.calls.filter(item => item.phase).length, 1);
  assert.equal(run.calls.at(-1), 'release');
  await run.controller.close();
}
{
  const run = fixture();
  await assert.rejects(run.stream({ ...base, maxRawOutputUtf16: 0 }).next(), /Raw UTF-16/);
  assert.equal(run.released.length, 1);
  assert.equal(run.calls.at(-1), 'release');
  await assert.rejects(run.controller.generateCanonicalTransitions({ ...plan, tokenSelection: {} }, request(base)).next(), /GPU token-selection/);
  await assert.rejects(run.controller.generateCanonicalTransitions(plan, { ...request(base), seed: 0 }).next(), /unknown contract field/);
}
console.log('canonical ordinary: semantic draws, C/L and empty admission, scratch-free execution, stops, barriers and cancellation passed');
