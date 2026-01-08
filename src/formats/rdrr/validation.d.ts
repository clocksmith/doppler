/**
 * RDRR Manifest Validation
 *
 * @module formats/rdrr/validation
 */

import type { RDRRManifest, ValidationResult } from './types.js';

export declare function validateManifest(manifest: Partial<RDRRManifest>): ValidationResult;
