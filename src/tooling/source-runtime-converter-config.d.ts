import type { ConverterConfigSchema } from '../config/schema/converter.schema.js';
import type { ManifestInferenceSchema } from '../config/schema/manifest.schema.js';
import type { ExecutionV1GraphSchema, ExecutionV1SessionSchema } from '../config/schema/execution-v1.schema.js';

export declare function createSourceRuntimeInference(rawConfig?: Record<string, unknown> | null): ManifestInferenceSchema;

export declare function createSourceRuntimeExecution(): ExecutionV1GraphSchema;

export declare function createSourceRuntimeSession(): ExecutionV1SessionSchema;

export declare function createSourceRuntimeConverterConfig(options?: {
  modelId?: string | null;
  rawConfig?: Record<string, unknown> | null;
  quantization?: Record<string, unknown> | null;
}): ConverterConfigSchema;
