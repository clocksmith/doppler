import { computeCanonicalSha256 as hash } from '../../src/formats/canonical-hash.js';
import { createForgeEvaluationSchedule } from '../../src/converter/forge-candidate-evaluation.js';

// Evaluation plumbing only: neither the source oracle nor metrics are physical.
export function createForgeEvaluationFixture(modelIRHash, candidateHashes) {
  const reference = { schema: 'doppler.forge-source-reference/v1', sourceHash: hash('fixture-source'),
    oracleHash: hash('fixture-oracle'), cases: [{ id: 'fixture', input: 'prompt', expected: { tokens: [1, 2, 3, 4] } }] };
  const contract = { schema: 'doppler.forge-candidate-evaluation/v1', evaluationId: 'fixture', modelIRHash, candidateHashes,
    referenceHash: hash(reference), scope: { surface: 'synthetic-webgpu', runtimeHash: hash('runtime'), environmentHash: hash('environment'),
      workloadHash: hash(reference.cases.map(({ id, input }) => ({ id, input }))), cacheMode: 'warm', loadMode: 'memory' },
    sampling: { warmupRuns: 1, timedRuns: 2, order: 'balanced-rotation', seed: 0 },
    checks: [{ id: 'tokens', mode: 'canonical-exact', maxAbsoluteError: null }],
    metrics: [{ id: 'latency', unit: 'ms', direction: 'minimize', limit: null }], selection: 'observed-range-pareto' };
  const observations = createForgeEvaluationSchedule(contract, reference).map(attempt => ({
    attemptId: attempt.attemptId, candidateHash: attempt.candidateHash, contractHash: hash(contract), scopeHash: hash(contract.scope),
    inputHash: attempt.inputHash, status: 'completed', output: { tokens: [1, 2, 3, 4] },
    metrics: attempt.phase === 'warmup' ? null : { latency: 2 - candidateHashes.indexOf(attempt.candidateHash) }, error: null,
  }));
  return { contract, reference, observations };
}
