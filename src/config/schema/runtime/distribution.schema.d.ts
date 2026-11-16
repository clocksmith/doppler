/**
 * Distribution Config Schema
 *
 * Configuration for network downloads, CDN settings, and retry policies.
 * These values control how model shards are fetched from remote servers.
 *
 * @module config/schema/distribution
 */

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

  /** Ordered list of shard delivery sources */
  sourceOrder: ('cache' | 'p2p' | 'http')[];

  /** Canonical source transition matrix for fallback behavior */
  sourceMatrix: {
    cache: {
      onHit: 'return';
      onMiss: 'next' | 'terminal';
      onFailure: 'next' | 'terminal';
    };
    p2p: {
      onHit: 'return';
      onMiss: 'next' | 'terminal';
      onFailure: 'next' | 'terminal';
    };
    http: {
      onHit: 'return';
      onMiss: 'next' | 'terminal';
      onFailure: 'next' | 'terminal';
    };
  };

  /** Deterministic source decision and trace policy */
  sourceDecision: {
    /** Keep routing deterministic for identical config/capabilities */
    deterministic: boolean;
    /** Optional per-shard decision trace */
    trace: {
      /** Emit source decision trace in shard delivery result */
      enabled: boolean;
      /** Include skipped sources in the trace plan */
      includeSkippedSources: boolean;
      /** Deterministic sampling ratio in [0, 1] for trace emission */
      samplingRate: number;
    };
  };

  /** Anti-rollback guard policy for cross-source fallback */
  antiRollback: {
    /** Enable guard checks for fallback transitions */
    enabled: boolean;
    /** Require expected shard hash before source attempts */
    requireExpectedHash: boolean;
    /** Require expected shard size before source attempts */
    requireExpectedSize: boolean;
    /** Require expected manifest version-set before source attempts */
    requireManifestVersionSet: boolean;
  };

  /** Optional P2P delivery configuration */
  p2p: {
    /** Enable P2P delivery attempts */
    enabled: boolean;
    /** Per-peer timeout in milliseconds */
    timeoutMs: number;
    /** Number of P2P retries before fallback */
    maxRetries: number;
    /** Delay between P2P retries in milliseconds */
    retryDelayMs: number;
    /** Versioned transport contract */
    contractVersion: number;
    /** Optional runtime-provided transport callback */
    transport: unknown;
    /** Optional control-plane callbacks for session + policy decisions */
    controlPlane: {
      /** Enable control-plane hooks */
      enabled: boolean;
      /** Versioned control-plane callback contract */
      contractVersion: number;
      /** Proactive token refresh skew in milliseconds */
      tokenRefreshSkewMs: number;
      /** Optional callback to issue/refresh session token */
      tokenProvider: unknown;
      /** Optional callback to allow/deny a shard request */
      policyEvaluator: unknown;
    };
    /** Feature-flagged browser WebRTC data-plane transport slice */
    webrtc: {
      /** Enable browser WebRTC data-plane transport */
      enabled: boolean;
      /** Optional static peer id */
      peerId: string | null;
      /** Data-channel request timeout in milliseconds */
      requestTimeoutMs: number;
      /** Max accepted payload size in bytes */
      maxPayloadBytes: number;
      /** Optional peer selector callback */
      selectPeer: unknown;
      /** Callback for obtaining open RTCDataChannel */
      getDataChannel: unknown;
    };
    /** Session/auth policy for P2P attempts */
    security: {
      /** Require a non-empty session token for P2P attempts */
      requireSessionToken: boolean;
      /** Session token value passed through runtime config */
      sessionToken: string | null;
      /** Optional absolute expiration timestamp (epoch ms) */
      tokenExpiresAtMs: number | null;
    };
    /** Runtime abuse controls for P2P attempts */
    abuse: {
      /** Max transport requests per minute (0 disables) */
      rateLimitPerMinute: number;
      /** Consecutive failures before temporary quarantine */
      maxConsecutiveFailures: number;
      /** Quarantine duration in milliseconds */
      quarantineMs: number;
    };
  };

  /** Minimum interval between progress callbacks in milliseconds */
  progressUpdateIntervalMs: number;

  /** Require a specific Content-Encoding for shard downloads (null disables check) */
  requiredContentEncoding: string | null;
}

/** Default distribution configuration */
export declare const DEFAULT_DISTRIBUTION_CONFIG: DistributionConfigSchema;
