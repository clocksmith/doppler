export function validateCallTimeOptions(options?: Record<string, unknown> | null): void;

export function validateRuntimeOverrides(overrides?: {
  inference?: {
    modelOverrides?: Record<string, unknown> | null;
  } | null;
} | null): void;
