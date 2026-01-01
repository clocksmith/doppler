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

/**
 * Configuration for network distribution and download behavior.
 *
 * Controls concurrent downloads, retry policies, chunk sizes, and CDN routing.
 * These settings affect network performance and reliability when fetching
 * model shards from remote servers.
 */
export interface DistributionConfigSchema {
  /** Number of concurrent shard downloads (1-8) */
  concurrentDownloads: number;

  /** Maximum retry attempts for failed downloads */
  maxRetries: number;

  /** Initial delay before first retry in milliseconds */
  initialRetryDelayMs: number;

  /** Maximum delay between retries in milliseconds (exponential backoff cap) */
  maxRetryDelayMs: number;

  /** Maximum chunk size for streaming downloads in bytes */
  maxChunkSizeBytes: number;

  /** CDN base path override (null uses origin server) */
  cdnBasePath: string | null;

  /** Minimum interval between progress callbacks in milliseconds */
  progressUpdateIntervalMs: number;
}

/** Default distribution configuration */
export const DEFAULT_DISTRIBUTION_CONFIG: DistributionConfigSchema = {
  concurrentDownloads: 3,
  maxRetries: 3,
  initialRetryDelayMs: 1000,
  maxRetryDelayMs: 30000,
  maxChunkSizeBytes: 8 * 1024 * 1024, // 8MB
  cdnBasePath: null,
  progressUpdateIntervalMs: 100,
};
