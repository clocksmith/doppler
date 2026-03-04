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
  sourceOrder: ['cache', 'p2p', 'http'],
  sourceMatrix: {
    cache: {
      onHit: 'return',
      onMiss: 'next',
      onFailure: 'next',
    },
    p2p: {
      onHit: 'return',
      onMiss: 'next',
      onFailure: 'next',
    },
    http: {
      onHit: 'return',
      onMiss: 'terminal',
      onFailure: 'terminal',
    },
  },
  sourceDecision: {
    deterministic: true,
    trace: {
      enabled: false,
      includeSkippedSources: true,
    },
  },
  antiRollback: {
    enabled: true,
    requireExpectedHash: true,
    requireExpectedSize: false,
    requireManifestVersionSet: true,
  },
  p2p: {
    enabled: false,
    timeoutMs: 3000,
    maxRetries: 1,
    retryDelayMs: 250,
    contractVersion: 1,
    transport: null,
  },
  progressUpdateIntervalMs: 100,
  requiredContentEncoding: null,
};
