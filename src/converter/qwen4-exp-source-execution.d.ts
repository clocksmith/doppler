import type { ExecutionV1GraphSchema } from '../config/schema/execution-v1.schema.js';
export declare function createQwen4ExpSourceExecution(
  tensors: ReadonlyArray<{ name: string; dtype: string }>, numLayers: number
): ExecutionV1GraphSchema;
