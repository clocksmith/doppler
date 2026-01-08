/**
 * Conversion Schema Definitions
 *
 * Types for model format conversion (GGUF/SafeTensors → RDRR).
 *
 * @module config/schema/conversion
 */

// =============================================================================
// Conversion Progress Schema
// =============================================================================

/** Conversion stages */
export const ConversionStage = {
  DETECTING: 'detecting',
  PARSING: 'parsing',
  QUANTIZING: 'quantizing',
  WRITING: 'writing',
  MANIFEST: 'manifest',
  COMPLETE: 'complete',
  ERROR: 'error',
};
