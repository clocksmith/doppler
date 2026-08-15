import assert from 'node:assert/strict';
import { computeCanonicalSha256 } from '../../src/utils/canonical-hash.js';
import {
  digestRegisteredVariantDescriptor,
} from '../../src/tooling/registered-variant-calibration.js';
import {
  runRegisteredVariantCalibrationJob,
} from '../../src/tooling/registered-variant-calibration-job.js';

const digest = (character) => `sha256:${character.repeat(64)}`;
const baselineDescriptor = {
  wgsl: 'matmul_baseline.wgsl',
  entryPoint: 'main',
  workgroup: [64, 1, 1],
  requires: [],
  inputDtype: 'f32',
  outputDtype: 'f32',
};
const candidateDescriptor = {
  wgsl: 'matmul_candidate.wgsl',
  entryPoint: 'main',
  workgroup: [64, 1, 1],
  requires: ['shader-f16'],
  inputDtype: 'f16',
  outputDtype: 'f16',
};
const baselineDescriptorDigest =
  digestRegisteredVariantDescriptor('matmul', 'baseline', baselineDescriptor);
const descriptorDigest =
  digestRegisteredVariantDescriptor('matmul', 'candidate', candidateDescriptor);
const baselineKernelDigest = digest('8');
const kernelDigest = digest('7');
const executionEngine = 'node:test';
const identity = {
  artifactDigest: digest('1'),
  manifestDigest: digest('2'),
  executionGraphDigest: digest('3'),
  executionEngineDigest: computeCanonicalSha256({
    surface: 'node',
    executionEngine,
  }),
  browserDigest: digest('4'),
  adapterDigest: digest('5'),
  wrapperDigest: digest('6'),
  capabilities: ['shader-f16'],
};
const registryId = 'test-matmul-candidate';
const kind = 'registered-kernel-variant';
const runtimeInputs = {
  runtimeProfile: null,
  runtimeConfig: { candidate: true },
};
const evidenceScope = {
  artifactDigest: identity.artifactDigest,
  executionGraphDigest: identity.executionGraphDigest,
  descriptorDigest,
  kernelDigest,
  executionEngineDigest: identity.executionEngineDigest,
  browserDigest: identity.browserDigest,
  adapterDigest: identity.adapterDigest,
};
const checkedInPath =
  'src/config/runtime/optimization-candidates/test-matmul-candidate.json';
const registryDigest = computeCanonicalSha256({
  registryId,
  kind,
  runtimeInputs,
  evidenceScope,
  checkedInPath,
});
const contract = {
  schema: 'doppler.runtime-optimization-contract/v1',
  contractId: 'test-real-variant',
  kind,
  campaign: {
    owner: 'doppler-runtime',
    changeClass: 'numerical-kernel',
    causalHypothesis: 'The registered f16 kernel improves decode throughput while preserving exact output.',
    expectedMetric: {
      path: 'result.metrics.decodeTokensPerSec',
      direction: 'maximize',
      minImprovementPercent: 5,
    },
    controlMetric: {
      path: 'result.output',
      expectation: 'unchanged',
    },
    endToEndAcceptanceMetric: {
      path: 'result.metrics.decodeTokensPerSec',
      direction: 'maximize',
      minImprovementPercent: 5,
    },
    budgets: {
      maxCandidates: 1,
      maxCommandRunsPerCandidate: 6,
    },
    stoppingRule: {
      kind: 'fixed-contract',
      retainNegativeResults: true,
    },
    retryConditions: ['The kernel, graph, provider, browser, or adapter identity changes.'],
    revocationConditions: ['Exact output parity or the declared throughput gate fails.'],
  },
  model: {
    modelId: 'test-model',
    modelUrl: null,
    expectedExecutionContractHash: null,
  },
  baseline: {
    runtimeProfile: null,
    runtimeConfig: { candidate: false },
  },
  workload: {
    type: 'inference',
    request: {
      inferenceInput: { prompt: 'hello', maxTokens: 128 },
      cacheMode: 'warm',
      loadMode: 'opfs',
    },
  },
  mutationPolicy: {
    references: [{ registryId, digest: registryDigest }],
    maxCandidates: 1,
  },
  verification: {
    comparisons: [{ path: 'result.output', mode: 'canonical_exact' }],
  },
  measurement: {
    metricPath: 'result.metrics.decodeTokensPerSec',
    direction: 'maximize',
    pairCount: 2,
    minValidPairs: 2,
    minImprovementPercent: 5,
    requirePositiveConfidence: true,
    maxRelativeStdDevPercent: 5,
  },
};
const binding = {
  artifactDigest: identity.artifactDigest,
  executionGraphDigest: identity.executionGraphDigest,
  descriptorDigest,
  kernelDigest,
  executionEngineDigest: identity.executionEngineDigest,
  browserDigest: identity.browserDigest,
  adapterDigest: identity.adapterDigest,
};
const job = {
  schema: 'doppler.registered-variant-calibration-job/v1',
  surface: 'node',
  plan: {
    schema: 'doppler.registered-variant-calibration-plan/v1',
    identity,
    baseline: {
      operation: 'matmul',
      variantId: 'baseline',
      descriptorDigest: baselineDescriptorDigest,
      kernelDigest: baselineKernelDigest,
    },
    candidates: [{
      operation: 'matmul',
      variantId: 'candidate',
      descriptorDigest,
      kernelDigest,
    }],
    shapeSuite: [{
      shapeId: 'decode-m1',
      phase: 'decode',
      sequenceLength: 128,
      batch: 1,
      heads: { query: 4, kv: 1, dim: 64 },
      tailClass: 'full-block',
      layouts: {
        input: 'row-major',
        weight: 'row-major',
        output: 'row-major',
        kv: 'contiguous',
      },
      dtypes: {
        storage: 'f16',
        materialization: 'f16',
        accumulation: 'f32',
      },
      fusionRole: 'projection',
      quantizationFormat: 'none',
    }],
  },
  correctnessEvidence: {
    'matmul/candidate': {
      binding,
      operatorReference: {
        'decode-m1': { passed: true, kernelDigest },
      },
      boundaryPack: {
        schema: 'doppler.boundary-comparison-receipt/v1',
        promotionGate: {
          boundaryCompatible: true,
          sourcePrecisionControlPassed: true,
        },
      },
      tokenParity: { exact: true, tokenCount: 128 },
    },
  },
  candidateRegistry: {
    schema: 'doppler.runtime-optimization-candidate-registry/v1',
    entries: {
      [registryId]: {
        registryId,
        kind,
        digest: registryDigest,
        runtimeInputs,
        evidenceScope,
        checkedInPath,
      },
    },
  },
  performance: {
    'matmul/candidate': { registryId, contract },
  },
};

const receipt = await runRegisteredVariantCalibrationJob(job, {
  registry: {
    operations: {
      matmul: {
        variants: {
          baseline: baselineDescriptor,
          candidate: candidateDescriptor,
        },
      },
    },
  },
  kernelDigests: {
    'matmul_baseline.wgsl#main': baselineKernelDigest,
    'matmul_candidate.wgsl#main': kernelDigest,
  },
  executionEngine,
  async runCommand(request) {
    const candidate = request.runtimeConfig.runtime.candidate === true;
    return {
      ok: true,
      result: {
        modelId: 'test-model',
        suite: request.command,
        passed: 1,
        failed: 0,
        output: 'same-output',
        metrics: {
          decodeTokensPerSec: candidate ? 110 : 100,
          executionContractArtifact: { stable: true },
          tokenCostLedger: {
            identity: {
              artifactDigest: identity.artifactDigest,
              executionGraphDigest: identity.executionGraphDigest,
              browserDigest: identity.browserDigest,
              adapterDigest: identity.adapterDigest,
            },
          },
        },
      },
    };
  },
});

assert.equal(receipt.executionSurface, 'node');
assert.equal(receipt.executionEngine, 'node:test');
assert.equal(receipt.results[0].decision, 'proposed');
assert.equal(
  receipt.results[0].proposal.selectionPolicy.afterPromotion,
  'required-on-compatible-hardware'
);
assert.equal(receipt.results[0].performance.decision.accepted, true);
assert.equal(receipt.runtimeMutationApplied, false);

await assert.rejects(
  () => runRegisteredVariantCalibrationJob(job, {
    registry: {
      operations: {
        matmul: {
          variants: {
            baseline: baselineDescriptor,
            candidate: candidateDescriptor,
          },
        },
      },
    },
    kernelDigests: {
      'matmul_baseline.wgsl#main': baselineKernelDigest,
      'matmul_candidate.wgsl#main': digest('8'),
    },
    executionEngine,
    async runCommand() {
      throw new Error('must not run');
    },
  }),
  /kernelDigest does not match current/
);

await assert.rejects(
  () => runRegisteredVariantCalibrationJob(job, {
    registry: {
      operations: {
        matmul: {
          variants: {
            baseline: baselineDescriptor,
            candidate: candidateDescriptor,
          },
        },
      },
    },
    kernelDigests: {
      'matmul_baseline.wgsl#main': baselineKernelDigest,
      'matmul_candidate.wgsl#main': kernelDigest,
    },
    executionEngine: 'node:different',
    async runCommand() {
      throw new Error('must not run');
    },
  }),
  /executionEngineDigest does not match/
);

console.log('registered-variant-calibration-job.test: ok');
