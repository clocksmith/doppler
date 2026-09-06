import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { computeCanonicalSha256 as hash } from '../../src/formats/canonical-hash.js';
import {
  createForgeEvaluationSchedule,
  evaluateForgeCandidates,
  runForgeCandidateEvaluation,
  validateForgeEvaluationContract,
} from '../../src/converter/forge-candidate-evaluation.js';

// Contract tests only: outputs and metrics are deterministic synthetic data,
// not retained physical measurements or proof of an implementation advantage.
const candidates = ['fast', 'small', 'dominated', 'incorrect', 'missing'].map(hash);
const reference = {
  schema: 'doppler.forge-source-reference/v1', sourceHash: hash('pinned-source'), oracleHash: hash('independent-oracle'),
  cases: [
    { id: 'short', input: { query: 'A', documents: ['B', 'C'] }, expected: { tokens: [1, 2], scores: [0.8, 0.2] } },
    { id: 'neighbor', input: { query: 'D', documents: ['E', 'F'] }, expected: { tokens: [3, 4], scores: [0.6, 0.4] } },
  ],
};
const contract = {
  schema: 'doppler.forge-candidate-evaluation/v1', evaluationId: 'synthetic-reranker-evaluation', modelIRHash: hash('model-ir'),
  candidateHashes: candidates, referenceHash: hash(reference),
  scope: { surface: 'synthetic-webgpu', runtimeHash: hash('runtime'), environmentHash: hash('environment'),
    workloadHash: hash(reference.cases.map(({ id, input }) => ({ id, input }))), cacheMode: 'warm', loadMode: 'memory' },
  sampling: { warmupRuns: 1, timedRuns: 3, order: 'balanced-rotation', seed: 5 },
  checks: [{ id: 'tokens', mode: 'canonical-exact', maxAbsoluteError: null },
    { id: 'scores', mode: 'absolute-array', maxAbsoluteError: 0.01 }],
  metrics: [{ id: 'latency', unit: 'ms', direction: 'minimize', limit: 100 },
    { id: 'memory', unit: 'bytes', direction: 'minimize', limit: null }],
  selection: 'observed-range-pareto',
};
const schema = JSON.parse(await fs.readFile('src/config/schema/forge-candidate-evaluation.schema.json', 'utf8'));
assert.deepEqual([...schema.required].sort(), Object.keys(contract).sort());
assert.equal(schema.properties.schema.const, contract.schema);
assert.equal(schema.additionalProperties, false);
assert.equal(validateForgeEvaluationContract(contract, reference), contract);

const schedule = createForgeEvaluationSchedule(contract, reference);
assert.equal(schedule.length, 5 * 4 * 2);
assert.deepEqual(createForgeEvaluationSchedule({ ...contract, candidateHashes: [...candidates].reverse() }, reference)
  .map(({ candidateHash, phase, run, caseId }) => ({ candidateHash, phase, run, caseId })),
schedule.map(({ candidateHash, phase, run, caseId }) => ({ candidateHash, phase, run, caseId })));
assert.notDeepEqual(schedule.slice(0, 5).map(row => row.candidateHash), schedule.slice(5, 10).map(row => row.candidateHash));

function observe(attempt, activeContract = contract) {
  const index = candidates.indexOf(attempt.candidateHash);
  const row = reference.cases.find(item => item.id === attempt.caseId);
  return { attemptId: attempt.attemptId, candidateHash: attempt.candidateHash,
    contractHash: hash(activeContract), scopeHash: hash(activeContract.scope), inputHash: attempt.inputHash,
    status: 'completed', output: structuredClone(row.expected),
    metrics: attempt.phase === 'warmup' ? null : { latency: [10, 20, 30, 1, 50][index], memory: [20, 10, 30, 1, 50][index] },
    error: null };
}
const observations = schedule.filter(attempt => attempt.candidateHash !== candidates[4]).map(attempt => {
  const result = observe(attempt);
  if (attempt.candidateHash === candidates[3] && attempt.phase === 'timed' && attempt.run === 2) result.output.scores[1] += 1;
  return result;
});
const before = JSON.stringify({ contract, reference, observations });
const result = evaluateForgeCandidates({ contract, reference, observations });
assert.deepEqual(result.selectedCandidateHashes, candidates.slice(0, 2).sort());
assert.equal(result.candidates.find(row => row.candidateHash === candidates[2]).dominatedBy.length, 2);
assert.equal(result.candidates.find(row => row.candidateHash === candidates[3]).eligible, false);
assert.ok(result.attempts.some(attempt => attempt.reason === 'missing_attempt'));
assert.equal(result.promotionAllowed, false);
assert.equal(result.claimAllowed, false);
assert.equal(JSON.stringify({ contract, reference, observations }), before);
assert.deepEqual(JSON.parse(JSON.stringify(result)), JSON.parse(JSON.stringify(evaluateForgeCandidates({ contract, reference, observations }))));

for (const change of [
  value => { value.metrics[0].limit = undefined; },
  value => { value.sampling.timedRuns = 0; },
  value => { value.sampling.seed = -1; },
  value => { value.sampling.order = 'candidate-grouped'; },
  value => { value.checks[0].maxAbsoluteError = 1; },
  value => { value.checks[1].maxAbsoluteError = -1; },
  value => { value.selection = 'proposal-score'; },
  value => { value.scope.workloadHash = hash('another-workload'); },
  value => { value.referenceHash = hash('another-reference'); },
  value => { value.candidateHashes.push(value.candidateHashes[0]); },
]) {
  const invalid = structuredClone(contract);
  change(invalid);
  assert.throws(() => validateForgeEvaluationContract(invalid, reference));
}
for (const change of [
  value => { value.scopeHash = hash('other-gpu'); },
  value => { value.contractHash = hash('weaker-policy'); },
  value => { value.candidateHash = candidates[1]; },
  value => { value.inputHash = hash('easier-input'); },
  value => { value.output.tokens[0] += 1; },
  value => { value.output.scores.pop(); },
  value => { value.output.extra = true; },
  value => { value.metrics.latency = 101; },
  value => { delete value.metrics.memory; },
  value => { value.metrics.latency = -1; },
  value => { value.error = 'device loss'; },
]) {
  const changed = structuredClone(observations);
  change(changed.find(row => schedule.find(attempt => attempt.attemptId === row.attemptId)?.phase === 'timed'
    && row.candidateHash === candidates[0]));
  const rejected = evaluateForgeCandidates({ contract, reference, observations: changed });
  assert.equal(rejected.candidates.find(row => row.candidateHash === candidates[0]).eligible, false);
}
for (const invalid of [[...observations, observations[0]], [...observations].reverse(), [{ attemptId: hash('unknown') }]]) {
  assert.throws(() => evaluateForgeCandidates({ contract, reference, observations: invalid }), /unknown, duplicate, or reordered/);
}
const nonfinite = structuredClone(observations);
nonfinite[0].output.scores[0] = NaN;
assert.throws(() => evaluateForgeCandidates({ contract, reference, observations: nonfinite }), /nonfinite/);

// Overlapping ranges are inconclusive; ties do not erase another implementation.
const allCorrect = schedule.map(attempt => observe(attempt));
for (const observation of allCorrect) {
  if (!observation.metrics) continue;
  observation.metrics = { latency: 20, memory: 20 };
}
assert.equal(evaluateForgeCandidates({ contract, reference, observations: allCorrect }).selectedCandidateHashes.length, 5);
const varied = structuredClone(allCorrect);
for (const observation of varied) {
  if (observation.candidateHash === candidates[0] && observation.metrics) {
    const attempt = schedule.find(row => row.attemptId === observation.attemptId);
    observation.metrics.latency = attempt.run === 2 ? 21 : 10;
  }
}
assert.equal(evaluateForgeCandidates({ contract, reference, observations: varied }).selectedCandidateHashes.length, 5);

// A neighboring case cannot be hidden by a better aggregate point estimate.
const neighborRegression = structuredClone(allCorrect);
for (const observation of neighborRegression) {
  if (observation.candidateHash === candidates[0] && observation.metrics) {
    const attempt = schedule.find(row => row.attemptId === observation.attemptId);
    observation.metrics.latency = attempt.caseId === 'neighbor' ? 21 : 1;
    observation.metrics.memory = 10;
  }
}
assert.equal(evaluateForgeCandidates({ contract, reference, observations: neighborRegression }).selectedCandidateHashes.length, 5);

const saved = [];
const episode = await runForgeCandidateEvaluation({ contract, reference,
  runAttempt: async ({ attempt }) => observe(attempt), onObservation: async observation => saved.push(observation) });
assert.equal(saved.length, schedule.length);
assert.deepEqual(episode.receipt, evaluateForgeCandidates(episode));
assert.deepEqual(episode.observations, saved);
const abort = new AbortController();
let calls = 0;
const cancelled = await runForgeCandidateEvaluation({ contract, reference, signal: abort.signal,
  runAttempt: async () => { calls += 1; abort.abort(); throw new Error('cancelled during readback'); } });
assert.equal(calls, 1);
assert.equal(cancelled.observations[0].status, 'cancelled');
assert.deepEqual(cancelled.receipt.selectedCandidateHashes, []);
assert.equal(cancelled.receipt.attempts.filter(row => row.reason === 'missing_attempt').length, schedule.length - 1);
const ignoredAbort = new AbortController();
const lateResult = await runForgeCandidateEvaluation({ contract, reference, signal: ignoredAbort.signal,
  runAttempt: async ({ attempt }) => { ignoredAbort.abort(); return observe(attempt); } });
assert.equal(lateResult.observations[0].status, 'cancelled', 'a late success cannot override cancellation');
assert.deepEqual(lateResult.receipt.selectedCandidateHashes, []);
const failure = await runForgeCandidateEvaluation({ contract, reference, runAttempt: async () => { throw new Error('device lost'); } });
assert.deepEqual(failure.receipt.selectedCandidateHashes, []);
assert.ok(failure.observations.every(row => row.status === 'failed' && row.error === 'device lost'));
await assert.rejects(runForgeCandidateEvaluation({ contract, reference, runAttempt: async ({ attempt }) => observe(attempt),
  onObservation: async () => { throw new Error('evidence storage failed'); } }), /evidence storage failed/);

console.log('✔ forge-candidate-evaluation.test.js passed (synthetic contract evidence only)');
