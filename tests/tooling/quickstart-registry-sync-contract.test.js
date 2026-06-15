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
