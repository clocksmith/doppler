export function matchesRule(match, context) {
  if (!match) return true;
  for (const [key, expected] of Object.entries(match)) {
    const actual = context[key];
    if (expected && typeof expected === 'object' && !Array.isArray(expected)) {
      if ('eq' in expected && actual !== expected.eq) return false;
      if ('neq' in expected && actual === expected.neq) return false;
      if ('gt' in expected && actual <= expected.gt) return false;
      if ('gte' in expected && actual < expected.gte) return false;
      if ('lt' in expected && actual >= expected.lt) return false;
      if ('lte' in expected && actual > expected.lte) return false;
      if ('in' in expected && !expected.in.includes(actual)) return false;
      continue;
    }
    if (actual !== expected) return false;
  }
  return true;
}

export function selectByRules(rules, context, fallback) {
  for (const rule of rules) {
    if (matchesRule(rule.match, context)) {
      return rule.value;
    }
  }
  return fallback;
}
