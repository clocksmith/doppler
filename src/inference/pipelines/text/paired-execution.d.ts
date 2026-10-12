import type { KernelPathSchema, RuntimeConfigSchema } from '../../../config/schema/index.js';
export declare function validatePairedExecution(
  path: KernelPathSchema | null,
  session: RuntimeConfigSchema['inference']['session'],
  manifest: import('./config.js').Manifest
): { activationPrecision: 'two-f32'; activationComponents: 2; activationElementStrideBytes: 8 };
