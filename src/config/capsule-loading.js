import schema from './capsule-loading.schema.json' with { type: 'json' };

export function normalizeCapsuleLoadingPolicy(options) {
  const policy = {};
  for (const [key, rule] of Object.entries(schema.properties)) {
    const value = options[key] === undefined ? rule.default : options[key];
    if (!(value === null && Array.isArray(rule.type) && rule.type.includes('null'))
      && (!Number.isSafeInteger(value) || value < rule.minimum || value > rule.maximum)) {
      throw new Error(`Invalid Capsule loading ${key}.`);
    }
    policy[key] = value;
  }
  return Object.freeze(policy);
}
