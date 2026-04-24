import { selectByRules } from '../gpu/kernels/rule-matcher.js';

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function matchesExactObject(actual, expected) {
  if (!isPlainObject(actual) || !isPlainObject(expected)) {
    return false;
  }
  const actualKeys = Object.keys(actual).sort();
  const expectedKeys = Object.keys(expected).sort();
  if (actualKeys.length !== expectedKeys.length) {
    return false;
  }
  for (let i = 0; i < actualKeys.length; i += 1) {
    if (actualKeys[i] !== expectedKeys[i]) {
      return false;
    }
  }
  for (const key of expectedKeys) {
    const expectedValue = expected[key];
    const actualValue = actual[key];
    if (isPlainObject(expectedValue)) {
      if (!matchesExactObject(actualValue, expectedValue)) {
        return false;
      }
      continue;
    }
    if (Array.isArray(expectedValue)) {
      if (!Array.isArray(actualValue) || actualValue.length !== expectedValue.length) {
        return false;
      }
      for (let i = 0; i < expectedValue.length; i += 1) {
        if (actualValue[i] !== expectedValue[i]) {
          return false;
        }
      }
      continue;
    }
    if (actualValue !== expectedValue) {
      return false;
    }
  }
  return true;
}

function decodeRecorderSemantic(context) {
  return context.hasDevice === true
    && context.debug !== true
    && context.disableCommandBatching !== true
    && context.kvLayout !== 'bdpa_paged';
}

function profileDecodeRecorderSemantic(context) {
  return context.hasDevice === true
    && context.debug !== true
    && context.kvLayout !== 'bdpa_paged';
}

function batchDecodeSemantic(context) {
  return context.batchSize > 1
    && context.useGPU === true
    && context.gpuSamplingAvailable === true
    && context.disableMultiTokenDecode !== true
    && context.disableCommandBatching !== true
    && context.isBdpaPagedLayout !== true
    && context.finitenessFallbackWindowOpen !== true
    && (
      context.hasLinearAttentionLayers !== true
      || context.hasRangeBackedPerLayerInputs !== true
    )
    && (
      context.hasRangeBackedPerLayerInputs !== true
      || context.selfSpeculationEnabled !== true
    );
}

function enumerateDecodeRecorderContexts() {
  const values = [true, false];
  const kvLayouts = ['bdpa_paged', 'paged', null];
  const contexts = [];
  for (const hasDevice of values) {
    for (const debug of values) {
      for (const disableCommandBatching of values) {
        for (const kvLayout of kvLayouts) {
          contexts.push({
            hasDevice,
            debug,
            disableCommandBatching,
            kvLayout,
          });
        }
      }
    }
  }
  return contexts;
}

function enumerateBatchDecodeContexts() {
  const values = [true, false];
  const batchSizes = [1, 2];
  const contexts = [];
  for (const batchSize of batchSizes) {
    for (const useGPU of values) {
      for (const gpuSamplingAvailable of values) {
        for (const disableMultiTokenDecode of values) {
          for (const disableCommandBatching of values) {
            for (const isBdpaPagedLayout of values) {
              for (const finitenessFallbackWindowOpen of values) {
                for (const hasLinearAttentionLayers of values) {
                  for (const selfSpeculationEnabled of values) {
                    for (const hasRangeBackedPerLayerInputs of values) {
                      contexts.push({
                        batchSize,
                        useGPU,
                        gpuSamplingAvailable,
                        disableMultiTokenDecode,
                        disableCommandBatching,
                        isBdpaPagedLayout,
                        finitenessFallbackWindowOpen,
                        hasLinearAttentionLayers,
                        selfSpeculationEnabled,
                        hasRangeBackedPerLayerInputs,
                      });
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return contexts;
}

function maxBatchDecodeTokensSemantic(context) {
  if (context.hasHotVocabularyBatchDecode === true) {
    return 1;
  }
  if (context.hasLinearAttentionLayers === true) {
    return 32;
  }
  return context.hasGpuSplitPerLayerInputs === true ? 8 : null;
}

function enumerateMaxBatchDecodeTokenContexts() {
  return [
    { hasHotVocabularyBatchDecode: true, hasGpuSplitPerLayerInputs: false, hasLinearAttentionLayers: false },
    { hasHotVocabularyBatchDecode: false, hasGpuSplitPerLayerInputs: false, hasLinearAttentionLayers: true },
    { hasHotVocabularyBatchDecode: false, hasGpuSplitPerLayerInputs: true, hasLinearAttentionLayers: false },
    { hasHotVocabularyBatchDecode: false, hasGpuSplitPerLayerInputs: false, hasLinearAttentionLayers: false },
  ];
}

function prefillRecorderChunkLayersSemantic(context) {
  if (context.hasGpuSplitPerLayerInputs === true && context.numTokens <= 32) {
    return 8;
  }
  return 4;
}

function enumeratePrefillRecorderChunkLayerContexts() {
  return [
    { hasGpuSplitPerLayerInputs: true, numTokens: 1 },
    { hasGpuSplitPerLayerInputs: true, numTokens: 15 },
    { hasGpuSplitPerLayerInputs: true, numTokens: 32 },
    { hasGpuSplitPerLayerInputs: true, numTokens: 33 },
    { hasGpuSplitPerLayerInputs: false, numTokens: 15 },
    { hasGpuSplitPerLayerInputs: false, numTokens: 64 },
  ];
}

function checkRuleShape(rules, expectedMatches, label) {
  if (!Array.isArray(rules)) {
    return {
      ok: false,
      errors: [`[ExecutionRulesContract] ${label} must be an array.`],
    };
  }
  if (!Array.isArray(expectedMatches) || expectedMatches.length === 0) {
    return {
      ok: false,
      errors: [`[ExecutionRulesContract] ${label} expectedMatches must be a non-empty array.`],
    };
  }
  if (rules.length !== expectedMatches.length + 1) {
    return {
      ok: false,
      errors: [
        `[ExecutionRulesContract] ${label} must contain exactly ${expectedMatches.length + 1} rules; got ${rules.length}.`,
      ],
    };
  }
  const errors = [];
  for (let index = 0; index < expectedMatches.length; index += 1) {
    const rule = rules[index];
    if (!matchesExactObject(rule?.match, expectedMatches[index]) || rule?.value !== true) {
      errors.push(
        `[ExecutionRulesContract] ${label} rule ${index + 1} drifted from the expected enabling predicate.`
      );
      break;
    }
  }
  const fallbackRule = rules[rules.length - 1];
  if (!matchesExactObject(fallbackRule?.match, {}) || fallbackRule?.value !== false) {
    errors.push(`[ExecutionRulesContract] ${label} fallback rule must be { match: {}, value: false }.`);
  }
  return {
    ok: errors.length === 0,
    errors,
  };
}

function checkRuleSemantics(rules, contexts, expectedValue, label) {
  const errors = [];
  for (const context of contexts) {
    const actual = selectByRules(rules, context);
    const expected = expectedValue(context);
    if (actual !== expected) {
      errors.push(
        `[ExecutionRulesContract] ${label} mismatched context ${JSON.stringify(context)}: ` +
        `expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}.`
      );
      break;
    }
  }
  return {
    ok: errors.length === 0,
    errors,
    sampledContexts: contexts.length,
  };
}

export function buildInferenceExecutionRulesContractArtifact(ruleGroup) {
  const errors = [];
  const checks = [];
  const decodeRules = ruleGroup?.decodeRecorderEnabled;
  const profileDecodeRules = ruleGroup?.profileDecodeRecorderEnabled;
  const batchRules = ruleGroup?.batchDecodeEnabled;
  const maxBatchDecodeTokenRules = ruleGroup?.maxBatchDecodeTokens;
  const prefillRecorderChunkLayerRules = ruleGroup?.prefillRecorderChunkLayers;

  const decodeShape = checkRuleShape(
    decodeRules,
    [{
      hasDevice: true,
      debug: false,
      disableCommandBatching: false,
      kvLayout: { neq: 'bdpa_paged' },
    }],
    'decodeRecorderEnabled'
  );
  errors.push(...decodeShape.errors);
  checks.push({
    id: 'inference.execution.decodeRecorderEnabled.shape',
    ok: decodeShape.ok,
  });

  const decodeSemantics = Array.isArray(decodeRules)
    ? checkRuleSemantics(
      decodeRules,
      enumerateDecodeRecorderContexts(),
      decodeRecorderSemantic,
      'decodeRecorderEnabled'
    )
    : { ok: false, errors: ['[ExecutionRulesContract] decodeRecorderEnabled is unavailable for semantic check.'], sampledContexts: 0 };
  errors.push(...decodeSemantics.errors);
  checks.push({
    id: 'inference.execution.decodeRecorderEnabled.semantics',
    ok: decodeSemantics.ok,
  });

  const profileDecodeShape = checkRuleShape(
    profileDecodeRules,
    [{
      hasDevice: true,
      debug: false,
      kvLayout: { neq: 'bdpa_paged' },
    }],
    'profileDecodeRecorderEnabled'
  );
  errors.push(...profileDecodeShape.errors);
  checks.push({
    id: 'inference.execution.profileDecodeRecorderEnabled.shape',
    ok: profileDecodeShape.ok,
  });

  const profileDecodeSemantics = Array.isArray(profileDecodeRules)
    ? checkRuleSemantics(
      profileDecodeRules,
      enumerateDecodeRecorderContexts(),
      profileDecodeRecorderSemantic,
      'profileDecodeRecorderEnabled'
    )
    : {
      ok: false,
      errors: ['[ExecutionRulesContract] profileDecodeRecorderEnabled is unavailable for semantic check.'],
      sampledContexts: 0,
    };
  errors.push(...profileDecodeSemantics.errors);
  checks.push({
    id: 'inference.execution.profileDecodeRecorderEnabled.semantics',
    ok: profileDecodeSemantics.ok,
  });

  const batchShape = checkRuleShape(
    batchRules,
    [{
      batchSize: { gt: 1 },
      useGPU: true,
      gpuSamplingAvailable: true,
      disableMultiTokenDecode: { neq: true },
      disableCommandBatching: false,
      isBdpaPagedLayout: false,
      finitenessFallbackWindowOpen: false,
      hasLinearAttentionLayers: false,
      hasRangeBackedPerLayerInputs: false,
    }, {
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
    }, {
      batchSize: { gt: 1 },
      useGPU: true,
      gpuSamplingAvailable: true,
      disableMultiTokenDecode: { neq: true },
      disableCommandBatching: false,
      isBdpaPagedLayout: false,
      finitenessFallbackWindowOpen: false,
      hasLinearAttentionLayers: true,
      hasRangeBackedPerLayerInputs: false,
    }],
    'batchDecodeEnabled'
  );
  errors.push(...batchShape.errors);
  checks.push({
    id: 'inference.execution.batchDecodeEnabled.shape',
    ok: batchShape.ok,
  });

  const batchSemantics = Array.isArray(batchRules)
    ? checkRuleSemantics(
      batchRules,
      enumerateBatchDecodeContexts(),
      batchDecodeSemantic,
      'batchDecodeEnabled'
    )
    : { ok: false, errors: ['[ExecutionRulesContract] batchDecodeEnabled is unavailable for semantic check.'], sampledContexts: 0 };
  errors.push(...batchSemantics.errors);
  checks.push({
    id: 'inference.execution.batchDecodeEnabled.semantics',
    ok: batchSemantics.ok,
  });

  const maxBatchShapeErrors = [];
  if (!Array.isArray(maxBatchDecodeTokenRules) || maxBatchDecodeTokenRules.length !== 4) {
    maxBatchShapeErrors.push('[ExecutionRulesContract] maxBatchDecodeTokens must contain exactly 4 rules.');
  } else {
    const [hotVocabularyRule, linearAttentionRule, splitTablesRule, fallbackRule] = maxBatchDecodeTokenRules;
    if (!matchesExactObject(hotVocabularyRule?.match, { hasHotVocabularyBatchDecode: true }) || hotVocabularyRule?.value !== 1) {
      maxBatchShapeErrors.push(
        '[ExecutionRulesContract] maxBatchDecodeTokens hot-vocabulary rule must cap tokenizer_scores bursts at 1 token.'
      );
    }
    if (!matchesExactObject(linearAttentionRule?.match, { hasLinearAttentionLayers: true }) || linearAttentionRule?.value !== 32) {
      maxBatchShapeErrors.push(
        '[ExecutionRulesContract] maxBatchDecodeTokens linear-attention rule must cap unsafe burst recording at 32 tokens.'
      );
    }
    if (!matchesExactObject(splitTablesRule?.match, { hasGpuSplitPerLayerInputs: true }) || splitTablesRule?.value !== 8) {
      maxBatchShapeErrors.push(
        '[ExecutionRulesContract] maxBatchDecodeTokens enabling rule must cap gpu_split_tables bursts at 8 tokens.'
      );
    }
    if (!matchesExactObject(fallbackRule?.match, {}) || fallbackRule?.value !== null) {
      maxBatchShapeErrors.push(
        '[ExecutionRulesContract] maxBatchDecodeTokens fallback rule must be { match: {}, value: null }.'
      );
    }
  }
  errors.push(...maxBatchShapeErrors);
  checks.push({
    id: 'inference.execution.maxBatchDecodeTokens.shape',
    ok: maxBatchShapeErrors.length === 0,
  });

  const maxBatchSemantics = Array.isArray(maxBatchDecodeTokenRules)
    ? checkRuleSemantics(
      maxBatchDecodeTokenRules,
      enumerateMaxBatchDecodeTokenContexts(),
      maxBatchDecodeTokensSemantic,
      'maxBatchDecodeTokens'
    )
    : {
      ok: false,
      errors: ['[ExecutionRulesContract] maxBatchDecodeTokens is unavailable for semantic check.'],
      sampledContexts: 0,
    };
  errors.push(...maxBatchSemantics.errors);
  checks.push({
    id: 'inference.execution.maxBatchDecodeTokens.semantics',
    ok: maxBatchSemantics.ok,
  });

  const prefillChunkShapeErrors = [];
  if (!Array.isArray(prefillRecorderChunkLayerRules) || prefillRecorderChunkLayerRules.length !== 2) {
    prefillChunkShapeErrors.push(
      '[ExecutionRulesContract] prefillRecorderChunkLayers must contain exactly 2 rules.'
    );
  } else {
    const [shortSplitTablesRule, fallbackRule] = prefillRecorderChunkLayerRules;
    if (
      !matchesExactObject(shortSplitTablesRule?.match, {
        hasGpuSplitPerLayerInputs: true,
        numTokens: { lte: 32 },
      })
      || shortSplitTablesRule?.value !== 8
    ) {
      prefillChunkShapeErrors.push(
        '[ExecutionRulesContract] prefillRecorderChunkLayers short-prompt gpu_split_tables rule must use 8-layer chunks.'
      );
    }
    if (!matchesExactObject(fallbackRule?.match, {}) || fallbackRule?.value !== 4) {
      prefillChunkShapeErrors.push(
        '[ExecutionRulesContract] prefillRecorderChunkLayers fallback rule must be { match: {}, value: 4 }.'
      );
    }
  }
  errors.push(...prefillChunkShapeErrors);
  checks.push({
    id: 'inference.execution.prefillRecorderChunkLayers.shape',
    ok: prefillChunkShapeErrors.length === 0,
  });

  const prefillChunkSemantics = Array.isArray(prefillRecorderChunkLayerRules)
    ? checkRuleSemantics(
      prefillRecorderChunkLayerRules,
      enumeratePrefillRecorderChunkLayerContexts(),
      prefillRecorderChunkLayersSemantic,
      'prefillRecorderChunkLayers'
    )
    : {
      ok: false,
      errors: ['[ExecutionRulesContract] prefillRecorderChunkLayers is unavailable for semantic check.'],
      sampledContexts: 0,
    };
  errors.push(...prefillChunkSemantics.errors);
  checks.push({
    id: 'inference.execution.prefillRecorderChunkLayers.semantics',
    ok: prefillChunkSemantics.ok,
  });

  return {
    schemaVersion: 1,
    source: 'doppler',
    ok: errors.length === 0,
    checks,
    errors,
    stats: {
      decodeRecorderRules: Array.isArray(decodeRules) ? decodeRules.length : 0,
      profileDecodeRecorderRules: Array.isArray(profileDecodeRules) ? profileDecodeRules.length : 0,
      batchDecodeRules: Array.isArray(batchRules) ? batchRules.length : 0,
      maxBatchDecodeTokenRules: Array.isArray(maxBatchDecodeTokenRules) ? maxBatchDecodeTokenRules.length : 0,
      prefillRecorderChunkLayerRules: Array.isArray(prefillRecorderChunkLayerRules) ? prefillRecorderChunkLayerRules.length : 0,
      decodeRecorderContexts: decodeSemantics.sampledContexts,
      profileDecodeRecorderContexts: profileDecodeSemantics.sampledContexts,
      batchDecodeContexts: batchSemantics.sampledContexts,
      maxBatchDecodeTokenContexts: maxBatchSemantics.sampledContexts,
      prefillRecorderChunkLayerContexts: prefillChunkSemantics.sampledContexts,
    },
  };
}
