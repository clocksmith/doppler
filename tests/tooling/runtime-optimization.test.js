import assert from 'node:assert/strict';
import {
  enumerateRuntimeOptimizationCandidates,
  evaluateBrowserRuntimeOptimizationCandidate,
  hashRuntimeOptimizationContract,
  materializeRuntimeOptimizationCandidate,
  validateRuntimeOptimizationContract,
  validateRuntimeOptimizationCandidateRegistry,
} from '../../src/tooling/runtime-optimization.js';
import { computeCanonicalSha256 } from '../../src/utils/canonical-hash.js';

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

{
  const runtimeInputs = {
    runtimeProfile: null,
    runtimeConfig: {
      inference: {
        executionPatch: {
          graphPatchId: 'checked-in-test-patch',
        },
      },
    },
  };
  const registryId = 'test-registered-kernel';
  const kind = 'registered-kernel-variant';
  const evidenceScope = {
    artifactDigest: `sha256:${'1'.repeat(64)}`,
    executionGraphDigest: `sha256:${'2'.repeat(64)}`,
  };
  const checkedInPath =
    'src/config/runtime/optimization-candidates/test-registered-kernel.json';
  const digest = computeCanonicalSha256({
    registryId,
    kind,
    runtimeInputs,
    evidenceScope,
    checkedInPath,
  });
  const registry = validateRuntimeOptimizationCandidateRegistry({
    schema: 'doppler.runtime-optimization-candidate-registry/v1',
    entries: {
      [registryId]: {
        registryId,
        kind,
        digest,
        runtimeInputs,
        evidenceScope,
        checkedInPath,
      },
    },
  });
  const contract = createContract({
    kind,
    mutationPolicy: {
      references: [{ registryId, digest }],
      maxCandidates: 1,
    },
  });
  const [candidate] = enumerateRuntimeOptimizationCandidates(contract);
  assert.equal(candidate.kind, kind);
  assert.deepEqual(
    materializeRuntimeOptimizationCandidate(contract, candidate, {
      candidateRegistry: registry,
    }),
    runtimeInputs
  );
  assert.throws(
    () => materializeRuntimeOptimizationCandidate(contract, candidate),
    /require candidateRegistry/
  );
}

{
  const contract = createContract();
  contract.measurement = {
    ...contract.measurement,
    pairCount: 6,
    minValidPairs: 2,
    orderPolicy: {
      kind: 'randomized-blocks',
      seed: 17,
      blockSize: 2,
    },
    sequentialDecision: {
      kind: 'bonferroni-fixed-looks',
      lookEveryPairs: 2,
      minimumPairs: 2,
      maximumLooks: 3,
      alpha: 0.05,
    },
  };
  const [candidate] = enumerateRuntimeOptimizationCandidates(contract);
  const receipt = await evaluateBrowserRuntimeOptimizationCandidate(contract, candidate, {
    runCommand: async (request) => responseFor(request),
  });
  assert.equal(receipt.decision.accepted, true);
  assert.equal(receipt.measurement.completedPairs, 2);
  assert.equal(receipt.measurement.sequentialDecision.stoppedEarly, true);
  assert.equal(receipt.measurement.sequentialDecision.stopDecision, 'accept');
  assert.notDeepEqual(
    receipt.measurement.pairs[0].order,
    receipt.measurement.pairs[1].order
  );
}

{
  const contract = createContract({
    neighboringWorkloads: [{
      guardId: 'prefill-neighbor',
      workload: {
        type: 'inference',
        request: {
          inferenceInput: { prompt: 'neighbor', maxTokens: 8 },
          cacheMode: 'warm',
          loadMode: 'opfs',
        },
      },
      metricPath: 'result.metrics.decodeTokensPerSec',
      direction: 'maximize',
      maxRegressionPercent: 5,
      pairCount: 1,
    }],
  });
  const [candidate] = enumerateRuntimeOptimizationCandidates(contract);
  const receipt = await evaluateBrowserRuntimeOptimizationCandidate(contract, candidate, {
    runCommand: async (request) => {
      const response = responseFor(request);
      const isNeighbor = request.inferenceInput?.prompt === 'neighbor';
      const batchSize = request.runtimeConfig.runtime.inference.session.decodeLoop.batchSize;
      if (isNeighbor && batchSize > 1 && request.command === 'bench') {
        response.result.metrics.decodeTokensPerSec = 80;
        response.result.timing = { decodeTokensPerSec: 80 };
      }
      return response;
    },
  });
  assert.equal(receipt.decision.accepted, false);
  assert.ok(receipt.decision.reasons.includes('neighboring_workload_guard_failed'));
  assert.equal(receipt.neighboringWorkloadGuards.results[0].passed, false);
}

console.log('runtime-optimization.test: ok');
