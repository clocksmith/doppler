import assert from 'node:assert/strict';
import {
  enumerateRuntimeOptimizationCandidates,
  evaluateBrowserRuntimeOptimizationCandidate,
  hashRuntimeOptimizationContract,
  materializeRuntimeOptimizationCandidate,
  validateRuntimeOptimizationContract,
} from '../../src/tooling/runtime-optimization.js';

function createContract(overrides = {}) {
  return {
    schema: 'doppler.runtime-optimization-contract/v1',
    contractId: 'qwen-decode-grid-v1',
    kind: 'runtime_profile',
    model: {
      modelId: 'qwen-test',
      modelUrl: null,
      expectedExecutionContractHash: null,
    },
    baseline: {
      runtimeProfile: null,
      runtimeConfig: {
        inference: {
          session: {
            decodeLoop: {
              batchSize: 1,
            },
          },
        },
      },
    },
    workload: {
      type: 'inference',
      request: {
        inferenceInput: { prompt: 'hello', maxTokens: 8 },
        cacheMode: 'warm',
        loadMode: 'opfs',
      },
    },
    mutationPolicy: {
      dimensions: [
        {
          path: '/inference/session/decodeLoop/batchSize',
          values: [2, 4],
        },
      ],
      maxCandidates: 4,
    },
    verification: {
      comparisons: [{ path: 'result.output', mode: 'canonical_exact' }],
    },
    measurement: {
      metricPath: 'result.metrics.decodeTokensPerSec',
      direction: 'maximize',
      pairCount: 3,
      minValidPairs: 3,
      minImprovementPercent: 1,
      requirePositiveConfidence: false,
      maxRelativeStdDevPercent: 20,
    },
    ...overrides,
  };
}

function responseFor(request, options = {}) {
  const batchSize = request.runtimeConfig.runtime.inference.session.decodeLoop.batchSize;
  const candidate = batchSize > 1;
  const output = options.candidateOutput && candidate ? options.candidateOutput : 'same output';
  const throughput = candidate ? 110 : 100;
  return {
    ok: true,
    schemaVersion: 1,
    surface: 'browser',
    request,
    result: {
      suite: request.command,
      passed: 1,
      failed: 0,
      skipped: 0,
      modelId: 'qwen-test',
      output,
      metrics: {
        decodeTokensPerSec: throughput,
        executionContractArtifact: {
          schemaVersion: 1,
          source: 'doppler',
          ok: true,
        },
      },
      timing: { decodeTokensPerSec: throughput },
      deviceInfo: { vendor: 'test' },
    },
  };
}

{
  const contract = validateRuntimeOptimizationContract(createContract());
  const candidates = enumerateRuntimeOptimizationCandidates(contract);
  assert.equal(candidates.length, 2);
  assert.equal(candidates[0].contractHash, hashRuntimeOptimizationContract(contract));
  assert.deepEqual(
    materializeRuntimeOptimizationCandidate(contract, candidates[1]).runtimeConfig
      .inference.session.decodeLoop,
    { batchSize: 4 }
  );
}

{
  const contract = createContract();
  contract.mutationPolicy.dimensions[0].path = '/shared/benchmark/run/timedRuns';
  assert.throws(
    () => validateRuntimeOptimizationContract(contract),
    /evaluator or manifest-owned policy/
  );
}

{
  const contract = createContract();
  const [candidate] = enumerateRuntimeOptimizationCandidates(contract);
  const requests = [];
  const receipt = await evaluateBrowserRuntimeOptimizationCandidate(contract, candidate, {
    runCommand: async (request) => {
      requests.push(request);
      return responseFor(request);
    },
  });
  assert.equal(receipt.decision.accepted, true);
  assert.equal(receipt.measurement.completedPairs, 3);
  assert.equal(receipt.measurement.improvementPercent.median, 10);
  assert.deepEqual(
    receipt.measurement.pairs.map((pair) => pair.order),
    [
      ['baseline', 'candidate'],
      ['candidate', 'baseline'],
      ['baseline', 'candidate'],
    ]
  );
  assert.equal(requests.length, 8);
  assert.ok(requests.every((request) => request.captureOutput === true));
}

{
  const contract = createContract();
  const [candidate] = enumerateRuntimeOptimizationCandidates(contract);
  let benchCalls = 0;
  const receipt = await evaluateBrowserRuntimeOptimizationCandidate(contract, candidate, {
    runCommand: async (request) => {
      if (request.command === 'bench') benchCalls += 1;
      return responseFor(request, { candidateOutput: 'different output' });
    },
  });
  assert.equal(receipt.decision.accepted, false);
  assert.deepEqual(receipt.decision.reasons, ['candidate_parity_failed']);
  assert.equal(benchCalls, 0);
}

console.log('runtime-optimization.test: ok');
