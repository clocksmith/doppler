/**
 * Feature Check - GPU capability checking utilities
 *
 * Provides utilities for checking device feature requirements.
 *
 * @module gpu/kernels/feature-check
 */

// ============================================================================
// Types
// ============================================================================

/** Minimum capabilities interface for feature checking */
export interface FeatureCapabilities {
  hasF16: boolean;
  hasSubgroups: boolean;
}

// ============================================================================
// Feature Checking
// ============================================================================

/**
 * Check if all required features are available
 */
export declare function hasRequiredFeatures(
  required: string[],
  capabilities: FeatureCapabilities
): boolean;

// ============================================================================
// Attention Validation
// ============================================================================

/**
 * Validate that attention parameters are within device limits
 */
export declare function validateAttentionLimits(
  seqLen: number,
  numHeads: number,
  headDim: number
): void;
