const frozenMatchEntries = new WeakMap();

function matchEntries(match) {
  if (!Object.isFrozen(match)) return Object.entries(match);
  let entries = frozenMatchEntries.get(match);
  if (entries === undefined) {
    const descriptors = Object.values(Object.getOwnPropertyDescriptors(match));
    entries = descriptors.every(descriptor => 'value' in descriptor) ? Object.entries(match) : null;
    frozenMatchEntries.set(match, entries);
  }
  return entries ?? Object.entries(match);
}

export function matchesRule(match, context) {
  if (!match) return true;
  for (const [key, expected] of matchEntries(match)) {
    const actual = context[key];
    if (expected && typeof expected === 'object' && !Array.isArray(expected)) {
      if ('eq' in expected && actual !== expected.eq) return false;
      if ('neq' in expected && actual === expected.neq) return false;
      if ('gt' in expected && actual <= expected.gt) return false;
      if ('gte' in expected && actual < expected.gte) return false;
      if ('lt' in expected && actual >= expected.lt) return false;
      if ('lte' in expected && actual > expected.lte) return false;
      if ('in' in expected && !expected.in.includes(actual)) return false;
      // String pattern matching
      if ('contains' in expected) {
        const patterns = Array.isArray(expected.contains) ? expected.contains : [expected.contains];
        if (!patterns.some(p => typeof actual === 'string' && actual.includes(p))) return false;
      }
      if ('startsWith' in expected) {
        const patterns = Array.isArray(expected.startsWith) ? expected.startsWith : [expected.startsWith];
        if (!patterns.some(p => typeof actual === 'string' && actual.startsWith(p))) return false;
      }
      if ('endsWith' in expected) {
        const patterns = Array.isArray(expected.endsWith) ? expected.endsWith : [expected.endsWith];
        if (!patterns.some(p => typeof actual === 'string' && actual.endsWith(p))) return false;
      }
      continue;
    }
    if (actual !== expected) return false;
  }
  return true;
}

export function selectByRules(rules, context) {
  for (const rule of rules) {
    if (matchesRule(rule.match, context)) {
      return rule.value;
    }
  }
  throw new Error(
    `RuleMatcher: no rule matched context ${JSON.stringify(context)}`
  );
}

// Grammar validation for rule groups. It checks structure and types only;
// match/value content is owned by the JSON rule maps.
const NUMERIC_OPERATORS = Object.freeze(['gt', 'gte', 'lt', 'lte']);
const SCALAR_OPERATORS = Object.freeze(['eq', 'neq']);
const SUBSTRING_OPERATORS = Object.freeze(['contains', 'startsWith', 'endsWith']);
const MATCH_OPERATORS = Object.freeze([...SCALAR_OPERATORS, ...NUMERIC_OPERATORS, 'in', ...SUBSTRING_OPERATORS]);
const MODEL_IDENTITY_KEYS = Object.freeze(['modelId', 'manifestModelType']);

function isRulePlainObject(value) {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function isScalar(value) {
  return value === null || ['string', 'number', 'boolean'].includes(typeof value);
}

function validateMatchCondition(key, condition, where) {
  if (isScalar(condition)) return [];
  if (!isRulePlainObject(condition)) {
    return [`${where}.match.${key} must be a scalar or an operator object.`];
  }
  const operators = Object.keys(condition);
  if (operators.length === 0) return [`${where}.match.${key} operator object is empty.`];
  const errors = [];
  for (const operator of operators) {
    const operand = condition[operator];
    if (!MATCH_OPERATORS.includes(operator)) {
      errors.push(`${where}.match.${key} uses unknown operator "${operator}".`);
    } else if (NUMERIC_OPERATORS.includes(operator) && !Number.isFinite(operand)) {
      errors.push(`${where}.match.${key}.${operator} must be a finite number.`);
    } else if (SCALAR_OPERATORS.includes(operator) && !isScalar(operand)) {
      errors.push(`${where}.match.${key}.${operator} must be a scalar.`);
    } else if (operator === 'in' && (!Array.isArray(operand) || operand.length === 0 || !operand.every(isScalar))) {
      errors.push(`${where}.match.${key}.in must be a non-empty array of scalars.`);
    } else if (SUBSTRING_OPERATORS.includes(operator) && MODEL_IDENTITY_KEYS.includes(key)) {
      errors.push(`${where}.match.${key} uses ${operator}; model identity must use exact values or explicit lists.`);
    }
  }
  return errors;
}

// spec: { contextKeys: string[], isValue(value): boolean, valueType: string, catchAll: boolean }
export function validateRuleGroupStructure(rules, spec, label) {
  if (!Array.isArray(rules) || rules.length === 0) {
    return [`${label} must be a non-empty rule array.`];
  }
  const errors = [];
  rules.forEach((rule, index) => {
    const where = `${label}[${index}]`;
    if (!isRulePlainObject(rule)) {
      errors.push(`${where} must be an object.`);
      return;
    }
    const extraKeys = Object.keys(rule).filter((key) => key !== 'match' && key !== 'value');
    if (extraKeys.length > 0) errors.push(`${where} has unknown keys: ${extraKeys.join(', ')}.`);
    if (!isRulePlainObject(rule.match)) {
      errors.push(`${where}.match must be an object.`);
    } else {
      for (const [key, condition] of Object.entries(rule.match)) {
        if (!spec.contextKeys.includes(key)) {
          errors.push(`${where}.match.${key} is not a context field (${spec.contextKeys.join(', ')}).`);
          continue;
        }
        errors.push(...validateMatchCondition(key, condition, where));
      }
      const isCatchAll = Object.keys(rule.match).length === 0;
      if (isCatchAll && index !== rules.length - 1) {
        errors.push(`${where} is a catch-all before the last rule; later rules are unreachable.`);
      }
    }
    if (!Object.hasOwn(rule, 'value') || !spec.isValue(rule.value)) {
      errors.push(`${where}.value must be ${spec.valueType}; got ${JSON.stringify(rule.value)}.`);
    }
  });
  const last = rules[rules.length - 1];
  if (spec.catchAll && !(isRulePlainObject(last?.match) && Object.keys(last.match).length === 0)) {
    errors.push(`${label} must end with a catch-all rule ({ "match": {} }).`);
  }
  return errors;
}

// Validates every group of one rule file against its structural specs and
// reports unknown or missing groups. Extra cross-group checks append errors.
export function buildRuleFileStructureArtifact(ruleFile, specs, { idPrefix, label, extraChecks = [] }) {
  const errors = [];
  const checks = [];
  const stats = {};
  if (!isRulePlainObject(ruleFile)) {
    errors.push(`[${label}] rule file must be an object.`);
  } else {
    const unknownGroups = Object.keys(ruleFile).filter((name) => !Object.hasOwn(specs, name));
    if (unknownGroups.length > 0) {
      errors.push(`[${label}] unknown rule groups: ${unknownGroups.join(', ')}.`);
    }
    checks.push({ id: `${idPrefix}.groups`, ok: unknownGroups.length === 0 });
  }
  for (const [name, spec] of Object.entries(specs)) {
    const rules = ruleFile?.[name];
    const groupErrors = validateRuleGroupStructure(rules, spec, name).map((message) => `[${label}] ${message}`);
    errors.push(...groupErrors);
    checks.push({ id: `${idPrefix}.${name}.structure`, ok: groupErrors.length === 0 });
    stats[`${name}Rules`] = Array.isArray(rules) ? rules.length : 0;
  }
  for (const check of extraChecks) {
    const checkErrors = check.run(ruleFile).map((message) => `[${label}] ${message}`);
    errors.push(...checkErrors);
    checks.push({ id: `${idPrefix}.${check.id}`, ok: checkErrors.length === 0 });
  }
  return {
    schemaVersion: 1,
    source: 'doppler',
    ok: errors.length === 0,
    checks,
    errors,
    stats,
  };
}
