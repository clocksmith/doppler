export function validateCallTimeOptions(options?: Record<string, unknown> | null): void;

export function validateRuntimeOverrides(overrides?: {
  inference?: {
    modelOverrides?: Record<string, unknown> | null;
  } | null;
} | null): void;

export function validateRuntimeConfig(runtimeConfig?: {
  shared?: {
    debug?: {
      pipeline?: { enabled?: boolean | null } | null;
      trace?: { enabled?: boolean | null } | null;
      logLevel?: { defaultLogLevel?: string | null } | null;
    } | null;
  } | null;
  loading?: {
    allowF32UpcastNonMatmul?: boolean | null;
  } | null;
  inference?: {
    compute?: {
      keepF32Weights?: boolean | null;
      activationDtype?: string | null;
    } | null;
  } | null;
} | null): void;
