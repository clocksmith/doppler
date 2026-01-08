/**
 * Distribution Config Schema
 *
 * Configuration for network downloads, CDN settings, and retry policies.
 * These values control how model shards are fetched from remote servers.
 *
 * @module config/schema/distribution
 */

// =============================================================================
// Distribution Config
// =============================================================================

/** Default distribution configuration */
export const DEFAULT_DISTRIBUTION_CONFIG = {
  concurrentDownloads: 3,
  maxRetries: 3,
  initialRetryDelayMs: 1000,
  maxRetryDelayMs: 30000,
  maxChunkSizeBytes: 8 * 1024 * 1024, // 8MB
  cdnBasePath: null,
  progressUpdateIntervalMs: 100,
};
