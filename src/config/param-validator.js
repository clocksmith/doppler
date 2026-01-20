import { log } from '../debug/index.js';
import { PARAM_CATEGORIES, CategoryRules } from './param-categories.js';
import { TOOLING_INTENTS, TOOLING_DIAGNOSTICS } from './schema/tooling.schema.js';

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

  const batching = runtimeConfig.inference?.batching;
  if (!batching) {
    throw new Error('DopplerConfigError: runtime.inference.batching is required.');
  }

  assertNullablePositiveInt('runtime.inference.batching.readbackInterval', batching.readbackInterval);
  assertNullablePositiveInt('runtime.inference.batching.ringTokens', batching.ringTokens);
  assertNullablePositiveInt('runtime.inference.batching.ringStop', batching.ringStop);
  assertNullablePositiveInt('runtime.inference.batching.ringStaging', batching.ringStaging);

  validateToolingIntent(runtimeConfig);

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

function validateToolingIntent(runtimeConfig) {
  const tooling = runtimeConfig.shared?.tooling;
  const intent = tooling?.intent ?? null;
  const diagnostics = tooling?.diagnostics ?? null;

  if (intent !== null && !TOOLING_INTENTS.includes(intent)) {
    throw new Error(
      `DopplerConfigError: runtime.shared.tooling.intent must be one of ` +
      `${TOOLING_INTENTS.join(', ')} or null.`
    );
  }

  if (diagnostics !== null && !TOOLING_DIAGNOSTICS.includes(diagnostics)) {
    throw new Error(
      `DopplerConfigError: runtime.shared.tooling.diagnostics must be one of ` +
      `${TOOLING_DIAGNOSTICS.join(', ')}.`
    );
  }

  if (intent !== 'calibrate') return;

  const debug = runtimeConfig.shared?.debug;
  const benchmarkRun = runtimeConfig.shared?.benchmark?.run;
  const violations = [];

  if (debug?.trace?.enabled) violations.push('runtime.shared.debug.trace.enabled');
  if (debug?.pipeline?.enabled) violations.push('runtime.shared.debug.pipeline.enabled');
  if (debug?.probes?.length) violations.push('runtime.shared.debug.probes');
  if (debug?.profiler?.enabled) violations.push('runtime.shared.debug.profiler.enabled');
  if (benchmarkRun?.debug) violations.push('runtime.shared.benchmark.run.debug');
  if (benchmarkRun?.profile) violations.push('runtime.shared.benchmark.run.profile');
  if (benchmarkRun?.captureMemoryTimeSeries) {
    violations.push('runtime.shared.benchmark.run.captureMemoryTimeSeries');
  }

  if (violations.length === 0) return;

  throw new Error(
    'DopplerConfigError: runtime.shared.tooling.intent="calibrate" forbids ' +
    'investigation instrumentation.\n' +
    `Disable ${violations.join(', ')} or set runtime.shared.tooling.intent="investigate".`
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

function isDebugMode(debug) {
  if (!debug) return false;
  if (debug.pipeline?.enabled) return true;
  if (debug.trace?.enabled) return true;
  const level = debug.logLevel?.defaultLogLevel;
  return level === 'debug' || level === 'verbose';
}

function assertNullablePositiveInt(label, value) {
  if (value === undefined) {
    throw new Error(`DopplerConfigError: ${label} is required.`);
  }
  if (value === null) return;
  if (!Number.isFinite(value) || value <= 0 || Math.floor(value) !== value) {
    throw new Error(`DopplerConfigError: ${label} must be a positive integer or null.`);
  }
}
