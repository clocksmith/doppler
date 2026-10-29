export type { WeightLoadErrorDetails } from '../../../loader/tensor-load-error.js';
export { annotateWeightLoadError } from '../../../loader/tensor-load-error.js';

export type AnnotatableError = Error & {
  details?: Record<string, unknown>;
  code?: string;
};

export declare function rewriteWeightLoadError(
  error: unknown,
  context?: {
    modelId?: string | null;
    deviceLossInfo?: Record<string, unknown> | null;
  }
): unknown;
