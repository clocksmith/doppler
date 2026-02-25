import type {
  ExecutionV0ConfigSchema,
  ExecutionV0SessionDefaultsSchema,
} from '../config/schema/execution-v0.schema.js';

export interface ExecutionV0FromKernelPathSchema {
  schema: string;
  sessionDefaults: ExecutionV0SessionDefaultsSchema;
  execution: ExecutionV0ConfigSchema['execution'];
}

export declare function buildExecutionV0FromKernelPath(
  kernelPathRef: string | Record<string, unknown> | null | undefined
): ExecutionV0FromKernelPathSchema | null;

