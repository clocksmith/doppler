import type { DistributedPlanValidationOptions } from './validation.js';

export declare function parseDistributedPlan(
  jsonString: string,
  options?: DistributedPlanValidationOptions
): Record<string, unknown>;
