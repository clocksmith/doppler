import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { gate, readSwitches, submitResult, validateSwitches } from '../server/api.js';

const TOKENS = [5, 6, 7];
const protocol = {
  schema: 'doppler.observatory.protocol/v1', id: 'device-benchmark', version: '0.1.0',
  workload: { modelId: 'm', prompt: 'p', observationPolicy: 'demo/always-on', warmRuns: 1, generation: {} },
  reference: { generatedTokenIds: TOKENS },
  eligibility: { requireVisibleTab: true, requirePolicyPerformanceRepresentative: true, requirePolicyDoesNotModifyExecution: true, requireIdenticalTokensAcrossRuns: true, requireReferenceMatchWhenPinned: true, requirePlainPathCheck: true, maxPlainPathDeviation: 0.15 },
};
const run = (tokens = TOKENS) => ({ generatedTokenIds: tokens, wallMs: 500, timeToFirstTokenMs: 40, decodeTokensPerSecond: 30, execution: { decodeMode: 'single_token' } });
const evidence = (overrides = {}) => ({
  schema: 'doppler.observatory.device-benchmark/v1', protocol: { id: 'device-benchmark', version: '0.1.0' },
  failure: null, tabHiddenDuringRun: false, policy: { performanceRepresentative: true, modifiesExecution: false },
  cold: run(), warm: [run()], plainPath: [run()],
  // A page can claim anything here; the server must not read it.
  verdict: { eligible: true, state: 'passed' },
  ...overrides,
});
const memoryStore = () => { const kept = []; return { kept, append: async (item) => { kept.push(item); return `r${kept.length}`; } }; };
const on = { 'results.upload': true };

test('every publication switch in the checked-in server file is off', async () => {
  const switches = await readSwitches(new URL('../server/switches.json', import.meta.url));
  assert.deepEqual(switches, { 'results.upload': false, 'rankings.community': false, voting: false });
});

test('a switch that is off or missing refuses the action', () => {
  assert.equal(gate({ voting: false }, 'voting').status, 403);
  assert.equal(gate({}, 'voting').body.code, 'voting_disabled');
  assert.equal(gate({ voting: true }, 'voting'), null);
  assert.throws(() => validateSwitches({ schema: 'doppler.observatory.server-switches/v1', switches: { voting: 'on' } }), /true or false/);
});

test('with uploads off, a valid result is refused and nothing is stored', async () => {
  const store = memoryStore();
  const reply = await submitResult({ switches: { 'results.upload': false }, protocol, store, evidence: evidence() });
  assert.equal(reply.status, 403);
  assert.equal(reply.body.code, 'results_upload_disabled');
  assert.equal(store.kept.length, 0);
});

test('with uploads on, the server re-judges and stores its own verdict', async () => {
  const store = memoryStore();
  const reply = await submitResult({ switches: on, protocol, store, evidence: evidence() });
  assert.equal(reply.status, 201);
  assert.equal(store.kept[0].verdict.state, 'passed');
});

test('a result the page called eligible is refused when the server finds it is not', async () => {
  const store = memoryStore();
  const wrong = [5, 6, 9];
  const reply = await submitResult({ switches: on, protocol, store, evidence: evidence({ cold: run(wrong), warm: [run(wrong)], plainPath: [run(wrong)] }) });
  assert.equal(reply.status, 422);
  assert.match(reply.body.reasons[0], /pinned reference/);
  assert.equal(store.kept.length, 0);
});

test('a result for another protocol version is refused whatever the switches say', async () => {
  const reply = await submitResult({ switches: on, protocol, store: memoryStore(), evidence: evidence({ protocol: { id: 'device-benchmark', version: '0.0.9' } }) });
  assert.equal(reply.body.code, 'protocol_version_mismatch');
});

test('uploads on with nowhere to keep results says so', async () => {
  const reply = await submitResult({ switches: on, protocol, store: null, evidence: evidence() });
  assert.equal(reply.body.code, 'no_result_store');
});

test('the browser flags file and the server file name the same guarded switches', async () => {
  const flags = JSON.parse(await readFile(new URL('../config/flags.json', import.meta.url), 'utf8')).flags;
  const guarded = Object.keys(flags).filter((name) => flags[name].enforcedBy === 'server').sort();
  const server = Object.keys(await readSwitches(new URL('../server/switches.json', import.meta.url))).sort();
  assert.deepEqual(guarded, server);
});
