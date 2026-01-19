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

export function validateRuntimeConfig(runtimeConfig) {
  if (!runtimeConfig) return;

  const debug = runtimeConfig.shared?.debug;
  const debugEnabled = isDebugMode(debug);
  const allowF32Upcast = runtimeConfig.loading?.allowF32UpcastNonMatmul === true;
  const keepF32Weights = runtimeConfig.inference?.compute?.keepF32Weights === true;
  const activationDtype = runtimeConfig.inference?.compute?.activationDtype;
  const usesF32Activation = activationDtype === 'f32';

  if (!debugEnabled && (allowF32Upcast || keepF32Weights || usesF32Activation)) {
    const flags = [];
    if (allowF32Upcast) flags.push('runtime.loading.allowF32UpcastNonMatmul');
    if (keepF32Weights) flags.push('runtime.inference.compute.keepF32Weights');
    if (usesF32Activation) flags.push('runtime.inference.compute.activationDtype=f32');
    throw new Error(
      'DopplerConfigError: F32 weights/activations are debug-only. ' +
      `Disable ${flags.join(', ')} or enable runtime.shared.debug.pipeline.enabled ` +
      'or runtime.shared.debug.trace.enabled (or set log level to debug/verbose).'
    );
  }
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

function isDebugMode(debug) {
  if (!debug) return false;
  if (debug.pipeline?.enabled) return true;
  if (debug.trace?.enabled) return true;
  const level = debug.logLevel?.defaultLogLevel;
  return level === 'debug' || level === 'verbose';
}
