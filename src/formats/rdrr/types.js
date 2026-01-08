/**
 * RDRR Format Types
 *
 * Core type definitions for the RDRR model format.
 *
 * @module formats/rdrr/types
 */

import {
  RDRR_VERSION as SCHEMA_VERSION,
  SHARD_SIZE as SCHEMA_SHARD_SIZE,
  TENSORS_FILENAME as SCHEMA_TENSORS_FILENAME,
} from '../../config/schema/index.js';

// =============================================================================
// Re-exports from Schema
// =============================================================================

export const RDRR_VERSION = SCHEMA_VERSION;
export const SHARD_SIZE = SCHEMA_SHARD_SIZE;
export const MANIFEST_FILENAME = 'manifest.json';
export const TENSORS_FILENAME = SCHEMA_TENSORS_FILENAME;
