/**
 * AttnConfig runtime validation.
 *
 * Provides a validateAttnConfig() helper that checks all required fields are
 * defined (not undefined) after construction in layer.js. Logs warnings for
 * missing fields rather than throwing, preserving backward compatibility.
 *
 * @module inference/pipelines/text/attention/attn-config
 */

import type { AttentionConfig } from './types.js';

/**
 * Validate an AttnConfig object at construction time.
 *
 * Checks that all required fields are not undefined. Logs a warning per
 * missing field. Returns true when all required fields are present.
 *
 * @param config - The attention config to validate.
 * @param label - Diagnostic label (e.g., "L0" for layer 0).
 * @returns True if all required fields are present.
 */
export declare function validateAttnConfig(config: AttentionConfig, label?: string): boolean;
