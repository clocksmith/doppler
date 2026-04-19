import type { ConverterConfigSchema } from '../config/schema/converter.schema.js';
import type { ManifestInferenceSchema } from '../config/schema/manifest.schema.js';
import type { ConverterManifestConfigSchema } from '../config/schema/converter.schema.js';

export declare function createSourceRuntimeInference(rawConfig?: Record<string, unknown> | null): ManifestInferenceSchema;

export declare function createSourceRuntimeManifestConfig(
  rawConfig?: Record<string, unknown> | null
): ConverterManifestConfigSchema;

export declare function createSourceRuntimeManifestInference(
  rawConfig?: Record<string, unknown> | null
): ManifestInferenceSchema;

export declare function createSourceRuntimeConverterConfig(options?: {
  modelId?: string | null;
  rawConfig?: Record<string, unknown> | null;
  quantization?: Record<string, unknown> | null;
}): ConverterConfigSchema;
