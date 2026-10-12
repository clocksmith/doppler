import type { ExecutionPlanState, StaticExecutionPlan } from '../execution-plan.js';
import type { TensorRepresentation } from '../../../../gpu/tensor.js';
export declare function getActivationRepresentation(
  state: { executionPlanState?: ExecutionPlanState },
  executionPlan?: Pick<StaticExecutionPlan, 'activationDtype' | 'activationPrecision' | 'activationComponents' | 'activationElementStrideBytes'> | null
): TensorRepresentation;
