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
  assert.equal(artifact.stats.batchDecodeContexts, 256);
  assert.ok(
    artifact.checks.some((entry) => entry.id === 'inference.execution.decodeRecorderEnabled.semantics' && entry.ok)
  );
  assert.ok(
    artifact.checks.some((entry) => entry.id === 'inference.execution.profileDecodeRecorderEnabled.semantics' && entry.ok)
  );
  assert.ok(
    artifact.checks.some((entry) => entry.id === 'inference.execution.batchDecodeEnabled.semantics' && entry.ok)
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
          hasRangeBackedPerLayerInputs: false,
        },
        value: true,
      },
      { match: {}, value: false },
    ],
  });

  assert.equal(artifact.ok, false);
  assert.ok(
    artifact.errors.some((message) =>
      message.includes('decodeRecorderEnabled first rule drifted')
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
          hasRangeBackedPerLayerInputs: false,
        },
        value: true,
      },
      { match: {}, value: false },
    ],
  });

  assert.equal(artifact.ok, false);
  assert.ok(
    artifact.errors.some((message) =>
      message.includes('profileDecodeRecorderEnabled first rule drifted')
    )
  );
}

console.log('execution-rules-contract-check.test: ok');
