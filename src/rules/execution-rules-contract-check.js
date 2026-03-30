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
    && context.finitenessFallbackWindowOpen !== true;
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
                contexts.push({
                  batchSize,
                  useGPU,
                  gpuSamplingAvailable,
                  disableMultiTokenDecode,
                  disableCommandBatching,
                  isBdpaPagedLayout,
                  finitenessFallbackWindowOpen,
                });
              }
            }
          }
        }
      }
    }
  }
  return contexts;
}

function checkRuleShape(rules, expectedFirstMatch, label) {
  if (!Array.isArray(rules)) {
    return {
      ok: false,
      errors: [`[ExecutionRulesContract] ${label} must be an array.`],
    };
  }
  if (rules.length !== 2) {
    return {
      ok: false,
      errors: [`[ExecutionRulesContract] ${label} must contain exactly 2 rules; got ${rules.length}.`],
    };
  }
  const [firstRule, secondRule] = rules;
  const errors = [];
  if (!matchesExactObject(firstRule?.match, expectedFirstMatch) || firstRule?.value !== true) {
    errors.push(`[ExecutionRulesContract] ${label} first rule drifted from the expected enabling predicate.`);
  }
  if (!matchesExactObject(secondRule?.match, {}) || secondRule?.value !== false) {
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

  const decodeShape = checkRuleShape(
    decodeRules,
    {
      hasDevice: true,
      debug: false,
      disableCommandBatching: false,
      kvLayout: { neq: 'bdpa_paged' },
    },
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
    {
      hasDevice: true,
      debug: false,
      kvLayout: { neq: 'bdpa_paged' },
    },
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
    {
      batchSize: { gt: 1 },
      useGPU: true,
      gpuSamplingAvailable: true,
      disableMultiTokenDecode: { neq: true },
      disableCommandBatching: false,
      isBdpaPagedLayout: false,
      finitenessFallbackWindowOpen: false,
    },
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
      decodeRecorderContexts: decodeSemantics.sampledContexts,
      profileDecodeRecorderContexts: profileDecodeSemantics.sampledContexts,
      batchDecodeContexts: batchSemantics.sampledContexts,
    },
  };
}
