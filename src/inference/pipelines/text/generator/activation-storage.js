import { resolveActiveExecutionPlan } from '../execution-plan.js';
import { tensorElementBytes } from '../../../../gpu/tensor.js';

// Buffer storage follows the bound execution plan, never the component dtype alone.
/** @type {import('./activation-storage.js').getActivationRepresentation} */
export function getActivationRepresentation(state, executionPlan) {
  const plan = executionPlan ?? (state.executionPlanState ? resolveActiveExecutionPlan(state) : null);
  const representation = {
    precision: plan?.activationPrecision,
    components: plan?.activationComponents,
    elementStrideBytes: plan?.activationElementStrideBytes,
  };
  if (plan) tensorElementBytes(plan.activationDtype, representation);
  return representation;
}
