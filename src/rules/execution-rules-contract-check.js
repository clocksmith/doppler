import { buildRuleFileStructureArtifact } from './rule-matcher.js';

// Structural contract for inference/execution.rules.json. Context keys mirror
// the selection contexts built in inference/pipelines/text/execution-plan.js;
// rule values are owned only by the JSON file.

const isBoolean = (value) => typeof value === 'boolean';
const isPositiveInteger = (value) => Number.isInteger(value) && value >= 1;

const DECODE_RECORDER_CONTEXT = Object.freeze(['hasDevice', 'debug', 'disableCommandBatching', 'kvLayout']);

const EXECUTION_RULE_SPECS = Object.freeze({
  decodeRecorderEnabled: {
    contextKeys: DECODE_RECORDER_CONTEXT,
    isValue: isBoolean,
    valueType: 'a boolean',
    catchAll: true,
  },
  profileDecodeRecorderEnabled: {
    contextKeys: DECODE_RECORDER_CONTEXT,
    isValue: isBoolean,
    valueType: 'a boolean',
    catchAll: true,
  },
  batchDecodeEnabled: {
    contextKeys: Object.freeze([
      'batchSize',
      'useGPU',
      'gpuSamplingAvailable',
      'disableMultiTokenDecode',
      'disableCommandBatching',
      'isBdpaPagedLayout',
      'finitenessFallbackWindowOpen',
      'hasLinearAttentionLayers',
      'selfSpeculationEnabled',
      'hasRangeBackedPerLayerInputs',
    ]),
    isValue: isBoolean,
    valueType: 'a boolean',
    catchAll: true,
  },
  maxBatchDecodeTokens: {
    contextKeys: Object.freeze([
      'hasHotVocabularyBatchDecode',
      'hasGpuSplitPerLayerInputs',
      'hasLinearAttentionLayers',
      'modelId',
      'activationDtype',
      'currentSeqLen',
      'maxDecodeTokens',
      'numLayers',
      'hiddenSize',
    ]),
    isValue: (value) => value === null || isPositiveInteger(value),
    valueType: 'null or a positive integer',
    catchAll: true,
  },
  prefillRecorderChunkLayers: {
    contextKeys: Object.freeze(['hasGpuSplitPerLayerInputs', 'numTokens']),
    isValue: isPositiveInteger,
    valueType: 'a positive integer',
    catchAll: true,
  },
  finitenessFallbackActivationDtype: {
    contextKeys: Object.freeze(['activationDtype']),
    isValue: (value) => value === 'f16' || value === 'f32',
    valueType: '"f16" or "f32"',
    catchAll: true,
  },
});

export function buildInferenceExecutionRulesContractArtifact(ruleGroup) {
  return buildRuleFileStructureArtifact(ruleGroup, EXECUTION_RULE_SPECS, {
    idPrefix: 'inference.execution',
    label: 'ExecutionRulesContract',
  });
}
