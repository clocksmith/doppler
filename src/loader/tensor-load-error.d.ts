export interface WeightLoadErrorDetails {
  [key: string]: unknown;
}

export declare function annotateWeightLoadError<T>(
  error: T,
  details?: WeightLoadErrorDetails
): T;

export declare function annotateTensorLoadError(
  error: unknown,
  name: string,
  location: any,
  details?: Record<string, unknown>
): Error;
