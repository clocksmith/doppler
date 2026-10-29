import assert from 'node:assert/strict';

import {
  buildInferenceExecutionRulesContractArtifact,
} from '../../src/rules/execution-rules-contract-check.js';
import { getInferenceExecutionRulesContractArtifact } from '../../src/rules/rule-registry.js';
import ruleBundle from '../../src/rules/generated/rule-bundle.json' with { type: 'json' };

const checkedIn = ruleBundle.files['inference/execution.rules.json'];

function withGroup(name, rules) {
  return { ...structuredClone(checkedIn), [name]: rules };
}

function assertRejected(ruleFile, fragment) {
  const artifact = buildInferenceExecutionRulesContractArtifact(ruleFile);
  assert.equal(artifact.ok, false, `expected rejection containing "${fragment}"`);
  assert.ok(
    artifact.errors.some((message) => message.includes(fragment)),
    `expected an error containing "${fragment}"; got ${JSON.stringify(artifact.errors)}`
  );
}

{
  const artifact = getInferenceExecutionRulesContractArtifact();
  assert.equal(artifact.ok, true, artifact.errors.join(' | '));
  assert.ok(artifact.checks.every((entry) => entry.ok));
  assert.deepEqual(artifact, buildInferenceExecutionRulesContractArtifact(checkedIn));
}

// Rule values are policy: changing a value of the right type stays valid.
{
  const retuned = withGroup(
    'prefillRecorderChunkLayers',
    checkedIn.prefillRecorderChunkLayers.map((rule) => ({ ...rule, value: rule.value + 1 }))
  );
  assert.equal(buildInferenceExecutionRulesContractArtifact(retuned).ok, true);
}

assertRejected(null, 'rule file must be an object');
assertRejected({ ...checkedIn, unknownGroup: [] }, 'unknown rule groups: unknownGroup');
assertRejected(withGroup('decodeRecorderEnabled', undefined), 'decodeRecorderEnabled must be a non-empty rule array');
assertRejected(withGroup('decodeRecorderEnabled', []), 'decodeRecorderEnabled must be a non-empty rule array');
assertRejected(
  withGroup('decodeRecorderEnabled', [{ match: { hasDevice: true }, value: true }]),
  'decodeRecorderEnabled must end with a catch-all rule'
);
assertRejected(
  withGroup('decodeRecorderEnabled', [{ match: {}, value: false }, { match: { hasDevice: true }, value: true }]),
  'decodeRecorderEnabled[0] is a catch-all before the last rule'
);
assertRejected(
  withGroup('decodeRecorderEnabled', [{ match: {}, value: 'yes' }]),
  'decodeRecorderEnabled[0].value must be a boolean'
);
assertRejected(
  withGroup('batchDecodeEnabled', [{ match: { batchSzie: { gt: 1 } }, value: true }, { match: {}, value: false }]),
  'batchDecodeEnabled[0].match.batchSzie is not a context field'
);
assertRejected(
  withGroup('batchDecodeEnabled', [{ match: { batchSize: { gt: '1' } }, value: true }, { match: {}, value: false }]),
  'batchDecodeEnabled[0].match.batchSize.gt must be a finite number'
);
assertRejected(
  withGroup('batchDecodeEnabled', [{ match: { batchSize: { above: 1 } }, value: true }, { match: {}, value: false }]),
  'uses unknown operator "above"'
);
assertRejected(
  withGroup('maxBatchDecodeTokens', [{ match: { modelId: { startsWith: 'gemma-4' } }, value: 8 }, { match: {}, value: null }]),
  'model identity must use exact values or explicit lists'
);
assertRejected(
  withGroup('maxBatchDecodeTokens', [{ match: { modelId: { in: [] } }, value: 8 }, { match: {}, value: null }]),
  'maxBatchDecodeTokens[0].match.modelId.in must be a non-empty array of scalars'
);
assertRejected(
  withGroup('maxBatchDecodeTokens', [{ match: {}, value: 0 }]),
  'maxBatchDecodeTokens[0].value must be null or a positive integer'
);
assertRejected(
  withGroup('prefillRecorderChunkLayers', [{ match: {}, value: null }]),
  'prefillRecorderChunkLayers[0].value must be a positive integer'
);
assertRejected(
  withGroup('finitenessFallbackActivationDtype', [{ match: {}, value: 'bf16' }]),
  'finitenessFallbackActivationDtype[0].value must be "f16" or "f32"'
);
assertRejected(
  withGroup('prefillRecorderChunkLayers', [{ match: {}, value: 4, note: 'x' }]),
  'prefillRecorderChunkLayers[0] has unknown keys: note'
);

console.log('execution-rules-contract-check.test: ok');
