import { selectByRules } from '../gpu/kernels/rule-matcher.js';
import { computeGlobalLayers } from '../config/schema/inference.schema.js';

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
    if (actualValue !== expectedValue) {
      return false;
    }
  }
  return true;
}

function expectedPatternKind(context) {
  if (context.patternType === 'alternating' && context.globalPattern === 'even') {
    return 'alternating_even';
  }
  if (context.patternType === 'alternating' && context.globalPattern === 'odd') {
    return 'alternating_odd';
  }
  if (context.patternType === 'every_n') {
    return 'every_n';
  }
  return null;
}

function expectedLayerType(context) {
  if (context.patternKind === 'alternating_even') {
    return context.isEven ? 'full_attention' : 'sliding_attention';
  }
  if (context.patternKind === 'alternating_odd') {
    return context.isEven ? 'sliding_attention' : 'full_attention';
  }
  if (context.patternKind === 'every_n') {
    return context.isStride ? 'full_attention' : 'sliding_attention';
  }
  return null;
}

function enumeratePatternKindContexts() {
  const patternTypes = ['alternating', 'every_n', 'custom', null];
  const globalPatterns = ['even', 'odd', 'every_n', null];
  const contexts = [];
  for (const patternType of patternTypes) {
    for (const globalPattern of globalPatterns) {
      contexts.push({ patternType, globalPattern });
    }
  }
  return contexts;
}

function enumerateLayerTypeContexts() {
  const patternKinds = ['alternating_even', 'alternating_odd', 'every_n'];
  const booleans = [true, false];
  const contexts = [];
  for (const patternKind of patternKinds) {
    for (const isEven of booleans) {
      for (const isStride of booleans) {
        contexts.push({ patternKind, isEven, isStride });
      }
    }
  }
  return contexts;
}

function checkRuleShape(rules, expected, label) {
  if (!Array.isArray(rules)) {
    return {
      ok: false,
      errors: [`[LayerPatternContract] ${label} must be an array.`],
    };
  }
  if (rules.length !== expected.length) {
    return {
      ok: false,
      errors: [`[LayerPatternContract] ${label} must contain exactly ${expected.length} rules; got ${rules.length}.`],
    };
  }
  const errors = [];
  for (let i = 0; i < expected.length; i += 1) {
    if (!matchesExactObject(rules[i]?.match, expected[i].match) || rules[i]?.value !== expected[i].value) {
      errors.push(`[LayerPatternContract] ${label} rule[${i}] drifted from the expected decision table.`);
      break;
    }
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
        `[LayerPatternContract] ${label} mismatched context ${JSON.stringify(context)}: ` +
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

function checkGlobalLayerSemantics() {
  const checks = [
    {
      id: 'inference.layerPattern.computeGlobalLayers.even',
      actual: computeGlobalLayers({ type: 'alternating', globalPattern: 'even' }, 6),
      expected: [0, 2, 4],
    },
    {
      id: 'inference.layerPattern.computeGlobalLayers.odd',
      actual: computeGlobalLayers({ type: 'alternating', globalPattern: 'odd' }, 6),
      expected: [1, 3, 5],
    },
    {
      id: 'inference.layerPattern.computeGlobalLayers.every_n_offset',
      actual: computeGlobalLayers({ type: 'every_n', period: 6, offset: 5 }, 12),
      expected: [5, 11],
    },
    {
      id: 'inference.layerPattern.computeGlobalLayers.every_n_negative_offset',
      actual: computeGlobalLayers({ type: 'every_n', period: 6, offset: -1 }, 12),
      expected: [5, 11],
    },
  ];
  const errors = [];
  const results = [];
  for (const entry of checks) {
    const ok = JSON.stringify(entry.actual) === JSON.stringify(entry.expected);
    results.push({ id: entry.id, ok });
    if (!ok) {
      errors.push(
        `[LayerPatternContract] ${entry.id} expected ${JSON.stringify(entry.expected)}, got ${JSON.stringify(entry.actual)}.`
      );
    }
  }
  return {
    checks: results,
    errors,
  };
}

export function buildLayerPatternContractArtifact(ruleGroup) {
  const errors = [];
  const checks = [];
  const patternKindRules = ruleGroup?.patternKind;
  const layerTypeRules = ruleGroup?.layerType;

  const patternKindShape = checkRuleShape(patternKindRules, [
    { match: { patternType: 'alternating', globalPattern: 'even' }, value: 'alternating_even' },
    { match: { patternType: 'alternating', globalPattern: 'odd' }, value: 'alternating_odd' },
    { match: { patternType: 'every_n' }, value: 'every_n' },
    { match: {}, value: null },
  ], 'patternKind');
  errors.push(...patternKindShape.errors);
  checks.push({ id: 'inference.layerPattern.patternKind.shape', ok: patternKindShape.ok });

  const patternKindSemantics = Array.isArray(patternKindRules)
    ? checkRuleSemantics(patternKindRules, enumeratePatternKindContexts(), expectedPatternKind, 'patternKind')
    : { ok: false, errors: ['[LayerPatternContract] patternKind is unavailable for semantic check.'], sampledContexts: 0 };
  errors.push(...patternKindSemantics.errors);
  checks.push({ id: 'inference.layerPattern.patternKind.semantics', ok: patternKindSemantics.ok });

  const layerTypeShape = checkRuleShape(layerTypeRules, [
    { match: { patternKind: 'alternating_even', isEven: true }, value: 'full_attention' },
    { match: { patternKind: 'alternating_even' }, value: 'sliding_attention' },
    { match: { patternKind: 'alternating_odd', isEven: false }, value: 'full_attention' },
    { match: { patternKind: 'alternating_odd' }, value: 'sliding_attention' },
    { match: { patternKind: 'every_n', isStride: true }, value: 'full_attention' },
    { match: { patternKind: 'every_n' }, value: 'sliding_attention' },
  ], 'layerType');
  errors.push(...layerTypeShape.errors);
  checks.push({ id: 'inference.layerPattern.layerType.shape', ok: layerTypeShape.ok });

  const layerTypeSemantics = Array.isArray(layerTypeRules)
    ? checkRuleSemantics(layerTypeRules, enumerateLayerTypeContexts(), expectedLayerType, 'layerType')
    : { ok: false, errors: ['[LayerPatternContract] layerType is unavailable for semantic check.'], sampledContexts: 0 };
  errors.push(...layerTypeSemantics.errors);
  checks.push({ id: 'inference.layerPattern.layerType.semantics', ok: layerTypeSemantics.ok });

  const globalLayerSemantics = checkGlobalLayerSemantics();
  errors.push(...globalLayerSemantics.errors);
  checks.push(...globalLayerSemantics.checks);

  return {
    schemaVersion: 1,
    source: 'doppler',
    ok: errors.length === 0,
    checks,
    errors,
    stats: {
      patternKindRules: Array.isArray(patternKindRules) ? patternKindRules.length : 0,
      layerTypeRules: Array.isArray(layerTypeRules) ? layerTypeRules.length : 0,
      patternKindContexts: patternKindSemantics.sampledContexts,
      layerTypeContexts: layerTypeSemantics.sampledContexts,
    },
  };
}
