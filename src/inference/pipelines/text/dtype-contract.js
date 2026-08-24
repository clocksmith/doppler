
import { log } from '../../../debug/index.js';

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

export { assertImplicitDtypeTransitionAllowed } from '../../../config/dtype-transition-contract.js';
