export interface WeightLoadErrorDetails {
  [key: string]: unknown;
}

export type AnnotatableError = Error & {
  details?: Record<string, unknown>;
  code?: string;
};

export declare function annotateWeightLoadError<T>(
  error: T,
  details?: WeightLoadErrorDetails
): T;

export declare function rewriteWeightLoadError(
  error: unknown,
  context?: {
    modelId?: string | null;
    deviceLossInfo?: Record<string, unknown> | null;
  }
): unknown;
