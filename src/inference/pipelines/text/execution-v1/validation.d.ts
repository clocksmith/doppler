import type { ExecutionV1GraphSchema } from '../../../../config/schema/execution-v1.schema.js';

export declare function applyExecutionPatch(
  execution: ExecutionV1GraphSchema,
  executionPatch: Record<string, unknown> | null | undefined
): ExecutionV1GraphSchema;
