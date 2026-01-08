/**
 * Kernel Path Schema
 *
 * Defines explicit, ordered kernel dispatch sequences for inference.
 * Replaces the implicit q4kStrategy/fusedFFNQ4K configuration.
 *
 * A kernel path is a complete specification of:
 * - Which kernels run
 * - In what order
 * - With what override constants
 * - With what entry points
 *
 * @module config/schema/kernel-path
 */

// =============================================================================
// Defaults
// =============================================================================

/** Default entry point */
export const DEFAULT_ENTRY = 'main';

/** Default input slot */
export const DEFAULT_INPUT = 'hidden_state';

/** Default output slot */
export const DEFAULT_OUTPUT = 'hidden_state';
