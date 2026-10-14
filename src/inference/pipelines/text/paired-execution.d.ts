import type { KernelPathSchema, RuntimeConfigSchema } from '../../../config/schema/index.js';
export declare function validatePairedExecution(
  path: KernelPathSchema | null,
  session: RuntimeConfigSchema['inference']['session'],
  manifest: import('./config.js').Manifest
): { activationPrecision: 'two-f32'; activationComponents: 2; activationElementStrideBytes: 8 };

/**
 * Require the paired mechanism closure for the manifest's declared linear
 * norm-gate activation (absent field = schema default `silu`), and reject
 * declared kernels whose registry gate activation disagrees.
 */
export declare function assertPairedLinearMechanismClosure(
  manifestInference: import('../../../config/schema/manifest.schema.js').ManifestInferenceSchema
): void;
