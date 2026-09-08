import definition from './generation-contract.json' with { type: 'json' };
import { freezeCapsuleV2 } from './capsule-v2.js';

export const GENERATION_CONTRACT = freezeCapsuleV2(definition);

export class GenerationError extends Error {
  constructor(kind, message, options = {}) {
    super(message, options);
    this.name = 'GenerationError';
    this.code = GENERATION_CONTRACT.errors[kind];
    if (!this.code) throw new Error('Unknown generation error kind.');
  }
}

function requireValue(ok, message) {
  if (!ok) throw new GenerationError('invalidRequest', message);
}

export function validateGenerationField(name, value, rule = GENERATION_CONTRACT.options[name]) {
  requireValue(rule, `Unknown generation field: ${name}.`);
  if (rule.type === 'array') {
    requireValue(Array.isArray(value) && (!rule.nonempty || value.length > 0), `Invalid ${name}: array required.`);
    for (const item of value) validateGenerationField(name, item, {
      type: rule.items, minimum: rule.minimum, nonempty: rule.nonemptyItems,
    });
    return;
  }
  const matches = rule.type === 'integer' ? Number.isSafeInteger(value)
    : rule.type === 'number' ? Number.isFinite(value) : typeof value === rule.type;
  requireValue(matches, `Invalid ${name}: ${rule.type} required.`);
  requireValue(!rule.nonempty || value.trim().length > 0, `Invalid ${name}: non-empty text required.`);
  requireValue((rule.minimum === undefined || value >= rule.minimum)
    && (rule.exclusiveMinimum === undefined || value > rule.exclusiveMinimum)
    && (rule.maximum === undefined || value <= rule.maximum), `Invalid ${name}: outside its supported range.`);
}

export function validateGenerationInput(input) {
  requireValue(input && typeof input === 'object' && !Array.isArray(input), 'Generation input must be an object.');
  requireValue(Object.keys(input).every(key => Object.hasOwn(GENERATION_CONTRACT.input, key)), 'Unknown generation input field.');
  requireValue(GENERATION_CONTRACT.inputExactlyOne.filter(key => Object.hasOwn(input, key)).length === 1,
    'generate requires exactly one prompt or promptTokens input.');
  for (const [key, value] of Object.entries(input)) validateGenerationField(key, value, GENERATION_CONTRACT.input[key]);
}

export function resolveGenerationOptions(options) {
  requireValue(options && typeof options === 'object' && !Array.isArray(options), 'Generation options must be an object.');
  requireValue(Object.keys(options).every(key => Object.hasOwn(GENERATION_CONTRACT.options, key)), 'Unknown generation option field.');
  const resolved = {};
  for (const [key, rule] of Object.entries(GENERATION_CONTRACT.options)) {
    const value = options[key] === undefined ? rule.default : options[key];
    if (value === undefined) {
      requireValue(!rule.required && !(rule.requiredWhenPositive && options[rule.requiredWhenPositive] > 0),
        `Capsule generation requires explicit ${key}.`);
      continue;
    }
    validateGenerationField(key, value, rule);
    resolved[key] = structuredClone(value);
  }
  return freezeCapsuleV2(resolved);
}
