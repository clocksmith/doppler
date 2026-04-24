import { createDopplerError, ERROR_CODES } from '../../../errors/doppler-error.js';
import { validateDistributedPlan } from './validation.js';

export function parseDistributedPlan(jsonString, options = {}) {
  let plan;
  try {
    plan = JSON.parse(jsonString);
  } catch (error) {
    throw createDopplerError(
      ERROR_CODES.DISTRIBUTED_PLAN_INVALID,
      `Failed to parse distributed plan JSON: ${error instanceof Error ? error.message : String(error)}`
    );
  }
  const validation = validateDistributedPlan(plan, options);
  if (!validation.valid) {
    throw createDopplerError(
      validation.code ?? ERROR_CODES.DISTRIBUTED_PLAN_INVALID,
      `Invalid distributed plan:\n  - ${validation.errors.join('\n  - ')}`
    );
  }
  return plan;
}

