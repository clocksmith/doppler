import type { WeightBuffer } from '../../../gpu/weight-buffer.js';

export declare function resolveDiffusionActivationDtype(
  runtime: { latent?: { dtype?: string } } | null | undefined
): 'f16' | 'f32';

export declare function expectDiffusionWeight<T>(weight: T | null | undefined, label: string): T;

export declare function normalizeDiffusionLocationDtype(
  dtype: string | null | undefined
): 'f16' | 'f32' | null;

export declare function normalizeDiffusionMatmulLocationDtype(
  dtype: string | null | undefined
): string | null;

export declare function inferDiffusionMatmulDtypeFromBuffer(
  weight: GPUBuffer | WeightBuffer | null | undefined,
  N: number,
  K: number,
  preferred: string | null | undefined
): string | null | undefined;
