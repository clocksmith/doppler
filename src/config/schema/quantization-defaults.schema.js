/**
 * Quantization Defaults Config Schema
 *
 * Default quantization settings for the model converter.
 * Controls the target precision for different weight groups when not explicitly specified.
 *
 * @module config/schema/quantization-defaults
 */

/** Default quantization configuration */
export const DEFAULT_QUANTIZATION_DEFAULTS = {
  visionDtype: 'f16',
  audioDtype: 'f16',
  projectorDtype: 'f16',
};
