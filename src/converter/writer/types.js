/**
 * RDRR Writer Type Definitions
 *
 * All interfaces and types used by the RDRR model format writer.
 * Types imported from config/schema for single source of truth.
 *
 * @module converter/writer/types
 */

import {
  SHARD_SIZE as SCHEMA_SHARD_SIZE,
  DEFAULT_STORAGE_ALIGNMENT_CONFIG,
} from '../../config/schema/index.js';

export const DEFAULT_SHARD_SIZE = SCHEMA_SHARD_SIZE;
export const ALIGNMENT = DEFAULT_STORAGE_ALIGNMENT_CONFIG.bufferAlignmentBytes;
