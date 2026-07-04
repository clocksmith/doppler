import assert from 'node:assert/strict';
import { buildQuickstartRegistryPayload } from '../../tools/sync-quickstart-registry.js';

function quickstartModel(overrides = {}) {
  return {
    modelId: 'verified-text-model',
    family: 'gemma3',
    modes: ['text'],
    aliases: ['verified-text'],
    quickstart: true,
    sortOrder: 1,
    hf: {
      repoId: 'Clocksmith/rdrr',
      revision: 'abc123',
      path: 'models/verified-text-model',
    },
    sourceCheckpointId: 'source/checkpoint',
    weightPackId: 'verified-text-model-wp-v1',
    manifestVariantId: 'verified-text-model-mv-v1',
    artifactCompleteness: 'complete',
    runtimePromotionState: 'manifest-owned',
    weightsRefAllowed: false,
    vendorBenchmark: {
      transformersjs: {
        repoId: 'onnx-community/verified-text-model-ONNX',
        dtype: 'q4f16',
      },
    },
    benchmarkEvidence: {
      status: 'benchmark-selected',
      localClaimLaneId: 'verified-text-model-rdrr',
      runtimeReport: 'reports/release-claims/verified-text-model/report.json',
      compareResult: 'benchmarks/vendors/results/compare_unit.json',
      summarySvg: 'benchmarks/vendors/results/verified-text-model.svg',
    },
    lifecycle: {
      availability: {
        hf: true,
      },
      status: {
        runtime: 'active',
        tested: 'verified',
      },
      tested: {
        result: 'pass',
        contracts: {
          executionContractOk: true,
        },
      },
    },
    ...overrides,
  };
}

{
  const payload = buildQuickstartRegistryPayload({ models: [quickstartModel()] });
  assert.equal(payload.models.length, 1);
  assert.equal(payload.models[0].modelId, 'verified-text-model');
  assert.equal(payload.models[0].artifactCompleteness, 'complete');
  assert.deepEqual(payload.models[0].vendorBenchmark, {
    transformersjs: {
      repoId: 'onnx-community/verified-text-model-ONNX',
      dtype: 'q4f16',
    },
  });
  assert.deepEqual(payload.models[0].benchmarkEvidence, {
    status: 'benchmark-selected',
    localClaimLaneId: 'verified-text-model-rdrr',
    runtimeReport: 'reports/release-claims/verified-text-model/report.json',
    compareResult: 'benchmarks/vendors/results/compare_unit.json',
    summarySvg: 'benchmarks/vendors/results/verified-text-model.svg',
  });
}

assert.throws(
  () => buildQuickstartRegistryPayload({
    models: [
      quickstartModel({
        lifecycle: {
          availability: { hf: false },
          status: { runtime: 'active', tested: 'verified' },
          tested: { result: 'pass', contracts: { executionContractOk: true } },
        },
      }),
    ],
  }),
  /lifecycle\.availability\.hf=true/
);

assert.throws(
  () => buildQuickstartRegistryPayload({
    models: [
      quickstartModel({
        lifecycle: {
          availability: { hf: true },
          status: { runtime: 'active', tested: 'none' },
          tested: { result: 'pass', contracts: { executionContractOk: true } },
        },
      }),
    ],
  }),
  /lifecycle\.status\.tested="verified"/
);

assert.throws(
  () => buildQuickstartRegistryPayload({
    models: [
      quickstartModel({
        lifecycle: {
          availability: { hf: true },
          status: { runtime: 'active', tested: 'verified' },
          tested: { result: 'pass', contracts: { executionContractOk: false } },
        },
      }),
    ],
  }),
  /executionContractOk=true/
);

assert.throws(
  () => buildQuickstartRegistryPayload({
    models: [
      quickstartModel({
        modes: ['diffusion'],
      }),
    ],
  }),
  /text or embedding mode/
);

console.log('quickstart-registry-sync-contract.test: ok');
