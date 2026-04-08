
import { log } from '../../../debug/index.js';

function normalizeDtype(value, label) {
  if (value == null) {
    return null;
  }
  const normalized = String(value).trim().toLowerCase();
  if (normalized !== 'f16' && normalized !== 'f32') {
    throw new Error(`[DtypeContract] ${label} must be "f16" or "f32"; got "${value}".`);
  }
  return normalized;
}

/**
 * Resolve activation dtype from all available sources.
 *
 * Checks the execution plan, the runtime config compute section, and (optionally)
 * the model config. Returns the first non-nullish value along with diagnostics
 * showing every source that was consulted.
 *
 * @param {object|null} executionPlanState - Compiled execution plan state (may be null before compilation).
 * @param {object|null} runtimeConfig - Full runtime config object.
 * @param {object|null} modelConfig - Parsed model config (optional, for manifest-declared dtype).
 * @returns {{ activationDtype: string|null, source: string, allSources: Record<string, string|null> }}
 */
export function resolveActivationDtype(executionPlanState, runtimeConfig, modelConfig) {
  const fromExecutionPlan = executionPlanState?.primaryPlan?.activationDtype ?? null;
  const fromRuntimeConfig = runtimeConfig?.inference?.compute?.activationDtype ?? null;
  const fromModelConfig = modelConfig?.activationDtype ?? null;

  const allSources = {
    executionPlan: fromExecutionPlan,
    runtimeConfig: fromRuntimeConfig,
    modelConfig: fromModelConfig,
  };

  if (fromExecutionPlan != null) {
    return { activationDtype: fromExecutionPlan, source: 'executionPlan', allSources };
  }
  if (fromRuntimeConfig != null) {
    return { activationDtype: fromRuntimeConfig, source: 'runtimeConfig', allSources };
  }
  if (fromModelConfig != null) {
    return { activationDtype: fromModelConfig, source: 'modelConfig', allSources };
  }
  return { activationDtype: null, source: 'none', allSources };
}

/**
 * Assert dtype consistency across all resolution paths.
 *
 * Compares the activationDtype declared in:
 *   1. The compiled execution plan (generator path)
 *   2. runtimeConfig.inference.compute.activationDtype (logits fallback path)
 *   3. The layer context activationDtype (derived from execution plan at build time)
 *
 * If any two sources disagree, a warning is logged with all three values.
 * This function never throws — it is a diagnostic-only assertion.
 *
 * @param {object|null} executionPlanState - Compiled execution plan state.
 * @param {object|null} runtimeConfig - Full runtime config object.
 * @param {object|null} layerContext - Layer context object (or null if not yet built).
 * @returns {{ consistent: boolean, values: Record<string, string|null> }}
 */
export function assertDtypeConsistency(executionPlanState, runtimeConfig, layerContext) {
  const fromExecutionPlan = executionPlanState?.primaryPlan?.activationDtype ?? null;
  const fromRuntimeConfig = runtimeConfig?.inference?.compute?.activationDtype ?? null;
  const fromLayerContext = layerContext?.activationDtype ?? null;

  const values = {
    executionPlan: fromExecutionPlan,
    runtimeConfig: fromRuntimeConfig,
    layerContext: fromLayerContext,
  };

  // Collect all non-null values and check whether they agree
  const defined = Object.entries(values).filter(([, v]) => v != null);
  if (defined.length <= 1) {
    // Zero or one source defined — nothing to compare
    return { consistent: true, values };
  }

  const uniqueValues = new Set(defined.map(([, v]) => v));
  const consistent = uniqueValues.size === 1;

  if (!consistent) {
    const details = defined.map(([k, v]) => `${k}="${v}"`).join(', ');
    log.warn(
      'DtypeContract',
      `activationDtype divergence detected across resolution paths: ${details}. ` +
      'The execution plan value takes precedence at runtime, but the other sources ' +
      'should agree to avoid subtle dtype mismatches.'
    );
  }

  return { consistent, values };
}

export function assertImplicitDtypeTransitionAllowed(options = {}) {
  const policy = options.executionPolicies?.dtypeTransition ?? null;
  if (policy !== 'require_cast_step') {
    return;
  }

  const fromDtype = normalizeDtype(options.fromDtype, 'fromDtype');
  const toDtype = normalizeDtype(options.toDtype, 'toDtype');
  if (!fromDtype || !toDtype || fromDtype === toDtype) {
    return;
  }

  const op = typeof options.op === 'string' && options.op.trim()
    ? options.op.trim()
    : 'operation';
  const detail = typeof options.detail === 'string' && options.detail.trim()
    ? ` ${options.detail.trim()}`
    : '';
  throw new Error(
    `[ExecutionV1] ${op} requires implicit dtype transition ${fromDtype} -> ${toDtype}.${detail} ` +
    `execution.policies.dtypeTransition="${policy}" requires an explicit cast step in manifest.inference.execution.`
  );
}
