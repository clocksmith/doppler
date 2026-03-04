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
      samplingRate: 1,
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
    controlPlane: {
      enabled: false,
      contractVersion: 1,
      tokenRefreshSkewMs: 5000,
      tokenProvider: null,
      policyEvaluator: null,
    },
    webrtc: {
      enabled: false,
      peerId: null,
      requestTimeoutMs: 2500,
      maxPayloadBytes: 67108864,
      selectPeer: null,
      getDataChannel: null,
    },
    security: {
      requireSessionToken: false,
      sessionToken: null,
      tokenExpiresAtMs: null,
    },
    abuse: {
      rateLimitPerMinute: 0,
      maxConsecutiveFailures: 3,
      quarantineMs: 30000,
    },
  },
  progressUpdateIntervalMs: 100,
  requiredContentEncoding: null,
};
