import type { ModelIRV2 } from '../config/model-ir-v2.js';

export const TENSOR_ROLE_CLOSURE_SCHEMA_ID: 'doppler.tensor-role-closure/v1';

export declare function createTensorRoleClosureReceipt(options: {
  modelIR: ModelIRV2;
  headers: Record<string, unknown>;
  policy: Record<string, unknown>;
}): Record<string, unknown>;
