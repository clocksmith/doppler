import { MB } from './units.schema.js';

// =============================================================================
// Distribution Config
// =============================================================================

export const DEFAULT_DISTRIBUTION_CONFIG = {
  concurrentDownloads: 3,
  maxRetries: 3,
  initialRetryDelayMs: 1000,
  maxRetryDelayMs: 30000,
  maxChunkSizeBytes: 8 * MB,
  cdnBasePath: null,
  progressUpdateIntervalMs: 100,
  requiredContentEncoding: null,
};
