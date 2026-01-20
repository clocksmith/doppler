/**
 * Tensor-Config Consistency Validator
 *
 * Validates that manifest config flags are consistent with the actual tensors present.
 */

export interface TensorConfigWarning {
  severity: 'error' | 'warning';
  code: string;
  message: string;
  suggestion?: string;
}

export interface TensorConfigValidationResult {
  valid: boolean;
  warnings: TensorConfigWarning[];
  errors: TensorConfigWarning[];
}

/**
 * Validate tensor-config consistency.
 *
 * Checks that config flags (like `postFeedforwardNorm`) match the actual
 * tensors present in the model. Prevents silent failures from misconfigured
 * presets.
 *
 * @param manifest - The RDRR manifest to validate
 * @returns Validation result with errors and warnings
 */
export function validateTensorConfigConsistency(
  manifest: object
): TensorConfigValidationResult;

/**
 * Format validation result for logging.
 *
 * @param result - The validation result to format
 * @returns Formatted string for console/log output
 */
export function formatValidationResult(
  result: TensorConfigValidationResult
): string;
