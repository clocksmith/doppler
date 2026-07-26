import assert from 'node:assert/strict';
import {
  calibrateRegisteredVariants,
  digestRegisteredVariantDescriptor,
} from '../../src/tooling/registered-variant-calibration.js';

const baselineDescriptor = {
  wgsl: 'matmul.wgsl',
  entryPoint: 'main',
  workgroup: [16, 16, 1],
  requires: [],
};
const candidateDescriptor = {
  wgsl: 'matmul_subgroup.wgsl',
  entryPoint: 'main',
  workgroup: [256, 1, 1],
  requires: ['subgroups'],
};
const incompatibleDescriptor = {
  wgsl: 'matmul_f16.wgsl',
  entryPoint: 'main',
  workgroup: [256, 1, 1],
  requires: ['shader-f16'],
};
const registry = {
  operations: {
    matmul: {
      variants: {
        baseline: baselineDescriptor,
        subgroup: candidateDescriptor,
        f16: incompatibleDescriptor,
      },
    },
  },
};
const digest = (value) => `sha256:${value.repeat(64)}`;
const ref = (variantId, descriptor) => ({
  operation: 'matmul',
  variantId,
  descriptorDigest: digestRegisteredVariantDescriptor('matmul', variantId, descriptor),
  kernelDigest: digest(variantId === 'baseline' ? '6' : (variantId === 'subgroup' ? '7' : '8')),
});
const plan = {
  schema: 'doppler.registered-variant-calibration-plan/v1',
  identity: {
    artifactDigest: digest('1'),
    manifestDigest: digest('2'),
    executionGraphDigest: digest('3'),
    browserDigest: digest('4'),
    adapterDigest: digest('5'),
    wrapperDigest: digest('9'),
    capabilities: ['subgroups'],
  },
  baseline: ref('baseline', baselineDescriptor),
  candidates: [
    ref('subgroup', candidateDescriptor),
    ref('f16', incompatibleDescriptor),
  ],
  shapeSuite: [{
    shapeId: 'decode-m1-tail',
    phase: 'decode',
    sequenceLength: 128,
    batch: 1,
    heads: { query: 8, kv: 4, dim: 128 },
    tailClass: 'tail',
    layouts: {
      input: 'row-major',
      weight: 'q4-k',
      output: 'row-major',
      kv: 'contiguous',
    },
    dtypes: {
      storage: 'q4-k',
      materialization: 'f16',
      accumulation: 'f32',
    },
    fusionRole: 'projection',
    quantizationFormat: 'q4-k',
  }],
};
const calls = [];
const receipt = await calibrateRegisteredVariants(plan, {
  registry,
  runCorrectness: async (input) => {
    calls.push(input.mode);
    if (input.mode === 'operator-reference') {
      return {
        passed: true,
        kernelDigest: input.candidate.reference.kernelDigest,
      };
    }
    if (input.mode === 'boundary-pack') {
      return {
        schema: 'doppler.boundary-comparison-receipt/v1',
        promotionGate: {
          boundaryCompatible: true,
          sourcePrecisionControlPassed: true,
        },
      };
    }
    return { exact: true, tokenCount: 128 };
  },
  evaluatePerformance: async ({ typedCandidate }) => ({
    schema: 'doppler.runtime-optimization-receipt/v1',
    candidate: typedCandidate,
    decision: { accepted: true },
  }),
});
assert.deepEqual(calls, ['operator-reference', 'boundary-pack', 'token-parity']);
assert.equal(receipt.results[0].decision, 'proposed');
assert.equal(receipt.results[0].proposal.activation, 'manual-promotion-required');
assert.equal(receipt.results[1].decision, 'incompatible');
assert.equal(receipt.runtimeMutationApplied, false);

await assert.rejects(
  () => calibrateRegisteredVariants(plan, {
    registry,
    runCorrectness: async (input) => {
      if (input.mode === 'operator-reference') {
        return {
          passed: true,
          kernelDigest: input.candidate.reference.kernelDigest,
        };
      }
      if (input.mode === 'boundary-pack') {
        return {
          schema: 'doppler.boundary-comparison-receipt/v1',
          promotionGate: {
            boundaryCompatible: true,
            sourcePrecisionControlPassed: true,
          },
        };
      }
      return { exact: true, tokenCount: 128 };
    },
    evaluatePerformance: async () => ({ decision: { accepted: true } }),
  }),
  /must return doppler\.runtime-optimization-receipt\/v1/
);

console.log('registered-variant-calibration.test: ok');
