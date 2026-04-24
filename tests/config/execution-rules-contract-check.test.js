import assert from 'node:assert/strict';

import {
  buildInferenceExecutionRulesContractArtifact,
} from '../../src/rules/execution-rules-contract-check.js';
import {
  getInferenceExecutionRulesContractArtifact,
} from '../../src/rules/rule-registry.js';

{
  const artifact = getInferenceExecutionRulesContractArtifact();
  assert.equal(artifact.ok, true);
  assert.equal(artifact.stats.decodeRecorderContexts, 24);
  assert.equal(artifact.stats.profileDecodeRecorderContexts, 24);
  assert.equal(artifact.stats.batchDecodeContexts, 1024);
  assert.equal(artifact.stats.maxBatchDecodeTokenContexts, 4);
  assert.equal(artifact.stats.prefillRecorderChunkLayerContexts, 6);
  assert.ok(
    artifact.checks.some((entry) => entry.id === 'inference.execution.decodeRecorderEnabled.semantics' && entry.ok)
  );
  assert.ok(
    artifact.checks.some((entry) => entry.id === 'inference.execution.profileDecodeRecorderEnabled.semantics' && entry.ok)
  );
  assert.ok(
    artifact.checks.some((entry) => entry.id === 'inference.execution.batchDecodeEnabled.semantics' && entry.ok)
  );
  assert.ok(
    artifact.checks.some((entry) => entry.id === 'inference.execution.maxBatchDecodeTokens.semantics' && entry.ok)
  );
  assert.ok(
    artifact.checks.some((entry) => entry.id === 'inference.execution.prefillRecorderChunkLayers.semantics' && entry.ok)
  );
}

{
  const artifact = buildInferenceExecutionRulesContractArtifact({
    decodeRecorderEnabled: [
      {
        match: {
          hasDevice: true,
          debug: false,
          disableCommandBatching: false,
          kvLayout: { neq: 'paged' },
        },
        value: true,
      },
      { match: {}, value: false },
    ],
    profileDecodeRecorderEnabled: [
      {
        match: {
          hasDevice: true,
          debug: false,
          kvLayout: { neq: 'bdpa_paged' },
        },
        value: true,
      },
      { match: {}, value: false },
    ],
    batchDecodeEnabled: [
      {
        match: {
          batchSize: { gt: 1 },
          useGPU: true,
          gpuSamplingAvailable: true,
          disableMultiTokenDecode: { neq: true },
          disableCommandBatching: false,
          isBdpaPagedLayout: false,
          finitenessFallbackWindowOpen: false,
          hasLinearAttentionLayers: false,
          hasRangeBackedPerLayerInputs: false,
        },
        value: true,
      },
      {
        match: {
          batchSize: { gt: 1 },
          useGPU: true,
          gpuSamplingAvailable: true,
          disableMultiTokenDecode: { neq: true },
          disableCommandBatching: false,
          isBdpaPagedLayout: false,
          finitenessFallbackWindowOpen: false,
          hasLinearAttentionLayers: false,
          hasRangeBackedPerLayerInputs: true,
          selfSpeculationEnabled: false,
        },
        value: true,
      },
      { match: {}, value: false },
    ],
    maxBatchDecodeTokens: [
      { match: { hasHotVocabularyBatchDecode: true }, value: 2 },
      { match: { hasLinearAttentionLayers: true }, value: 64 },
      { match: { hasGpuSplitPerLayerInputs: true }, value: 8 },
      { match: {}, value: null },
    ],
    prefillRecorderChunkLayers: [
      {
        match: {
          hasGpuSplitPerLayerInputs: true,
          numTokens: { lte: 32 },
        },
        value: 8,
      },
      { match: {}, value: 4 },
    ],
  });

  assert.equal(artifact.ok, false);
  assert.ok(
    artifact.errors.some((message) =>
      message.includes('decodeRecorderEnabled rule 1 drifted')
    )
  );
}

{
  const artifact = buildInferenceExecutionRulesContractArtifact({
    decodeRecorderEnabled: [
      {
        match: {
          hasDevice: true,
          debug: false,
          disableCommandBatching: false,
          kvLayout: { neq: 'bdpa_paged' },
        },
        value: true,
      },
      { match: {}, value: false },
    ],
    profileDecodeRecorderEnabled: [
      {
        match: {
          hasDevice: true,
          debug: false,
          kvLayout: { neq: 'paged' },
        },
        value: true,
      },
      { match: {}, value: false },
    ],
    batchDecodeEnabled: [
      {
        match: {
          batchSize: { gt: 1 },
          useGPU: true,
          gpuSamplingAvailable: true,
          disableMultiTokenDecode: { neq: true },
          disableCommandBatching: false,
          isBdpaPagedLayout: false,
          finitenessFallbackWindowOpen: true,
          hasLinearAttentionLayers: false,
          hasRangeBackedPerLayerInputs: false,
        },
        value: true,
      },
      {
        match: {
          batchSize: { gt: 1 },
          useGPU: true,
          gpuSamplingAvailable: true,
          disableMultiTokenDecode: { neq: true },
          disableCommandBatching: false,
          isBdpaPagedLayout: false,
          finitenessFallbackWindowOpen: false,
          hasLinearAttentionLayers: false,
          hasRangeBackedPerLayerInputs: true,
          selfSpeculationEnabled: false,
        },
        value: true,
      },
      { match: {}, value: false },
    ],
    maxBatchDecodeTokens: [
      { match: { hasHotVocabularyBatchDecode: true }, value: 1 },
      { match: { hasLinearAttentionLayers: true }, value: 64 },
      { match: { hasGpuSplitPerLayerInputs: true }, value: 8 },
      { match: {}, value: 8 },
    ],
    prefillRecorderChunkLayers: [
      {
        match: {
          hasGpuSplitPerLayerInputs: true,
          numTokens: { lte: 32 },
        },
        value: 8,
      },
      { match: {}, value: 4 },
    ],
  });

  assert.equal(artifact.ok, false);
  assert.ok(
    artifact.errors.some((message) =>
      message.includes('profileDecodeRecorderEnabled rule 1 drifted')
    )
  );
  assert.ok(
    artifact.errors.some((message) =>
      message.includes('maxBatchDecodeTokens fallback rule must be { match: {}, value: null }.')
    )
  );
}

console.log('execution-rules-contract-check.test: ok');
