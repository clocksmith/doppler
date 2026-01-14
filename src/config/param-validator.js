import { log } from '../debug/index.js';
import { PARAM_CATEGORIES, CategoryRules } from './param-categories.js';

export function validateCallTimeOptions(options) {
  if (!options) return;

  const violations = [];
  for (const [key, value] of Object.entries(options)) {
    if (value === undefined) continue;

    const category = PARAM_CATEGORIES[key];
    if (!category) continue;

    if (!CategoryRules[category].callTime) {
      violations.push({ param: key, category });
    }
  }

  if (violations.length === 0) return;

  const violation = violations[0];
  const guidance = violation.category === 'model'
    ? 'Set via runtime.inference.modelOverrides (experimental) or manifest.'
    : 'Set via setRuntimeConfig() before generation.';

  throw new Error(
    `DopplerConfigError: "${violation.param}" is a ${violation.category} param. ` +
    'Cannot override at call-time.\n' +
    guidance
  );
}

export function validateRuntimeOverrides(overrides) {
  const modelOverrides = overrides?.inference?.modelOverrides;
  if (!modelOverrides) return;

  const params = flattenObject(modelOverrides);
  if (params.length === 0) return;

  log.warn(
    'Config',
    `Experimental: Overriding ${params.length} model param(s) via runtime: ${params.join(', ')}. ` +
      'Manifest values are recommended.'
  );
}

function flattenObject(obj, prefix = '') {
  const result = [];
  for (const [key, value] of Object.entries(obj)) {
    if (value === undefined || value === null) continue;
    const path = prefix ? `${prefix}.${key}` : key;
    if (typeof value === 'object' && !Array.isArray(value)) {
      result.push(...flattenObject(value, path));
    } else {
      result.push(path);
    }
  }
  return result;
}
