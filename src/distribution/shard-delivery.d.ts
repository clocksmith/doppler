export type DistributionSource = 'cache' | 'p2p' | 'http';

export interface DistributionShardInfo {
  filename?: string;
  size?: number;
  hash?: string;
  [key: string]: unknown;
}

export interface DistributionProgress {
  shardIndex: number;
  receivedBytes: number;
  totalBytes: number;
  percent: number;
}

export interface P2PTransportContext {
  shardIndex: number;
  shardInfo: DistributionShardInfo;
  signal?: AbortSignal;
  source: 'p2p';
  timeoutMs: number;
  contractVersion: number;
  attempt: number;
  maxRetries: number;
  resumeOffset?: number;
  expectedHash?: string | null;
  expectedSize?: number | null;
  expectedManifestVersionSet?: string | null;
}

export type P2PTransportResult =
  | ArrayBuffer
  | Uint8Array
  | {
    data?: ArrayBuffer | Uint8Array;
    buffer?: ArrayBuffer | Uint8Array;
    manifestVersionSet?: string | null;
    manifestHash?: string | null;
    rangeStart?: number | null;
    offset?: number | null;
    totalSize?: number | null;
    miss?: boolean;
    notFound?: boolean;
    error?: unknown;
  }
  | null
  | undefined;

export type P2PTransport = (
  context: P2PTransportContext
) => Promise<P2PTransportResult> | P2PTransportResult;

export interface DistributionP2PControlPlaneConfig {
  enabled?: boolean;
  contractVersion?: number;
  tokenRefreshSkewMs?: number;
  tokenProvider?: ((context: Record<string, unknown>) => Promise<unknown> | unknown) | null;
  policyEvaluator?: ((context: Record<string, unknown>) => Promise<unknown> | unknown) | null;
}

export interface DistributionP2PWebRTCConfig {
  enabled?: boolean;
  peerId?: string | null;
  requestTimeoutMs?: number;
  maxPayloadBytes?: number;
  selectPeer?: ((context: P2PTransportContext) => Promise<unknown> | unknown) | null;
  getDataChannel?: ((context: Record<string, unknown>) => Promise<unknown> | unknown) | null;
}

export interface DistributionP2PConfig {
  enabled?: boolean;
  timeoutMs?: number;
  maxRetries?: number;
  retryDelayMs?: number;
  contractVersion?: number;
  transport?: P2PTransport | null;
  controlPlane?: DistributionP2PControlPlaneConfig;
  webrtc?: DistributionP2PWebRTCConfig;
  security?: {
    requireSessionToken?: boolean;
    sessionToken?: string | null;
    tokenExpiresAtMs?: number | null;
  };
  abuse?: {
    rateLimitPerMinute?: number;
    maxConsecutiveFailures?: number;
    quarantineMs?: number;
  };
}

export interface DistributionSourceDecisionTraceConfig {
  enabled?: boolean;
  includeSkippedSources?: boolean;
  samplingRate?: number;
}

export interface DistributionSourceDecisionConfig {
  deterministic?: boolean;
  trace?: DistributionSourceDecisionTraceConfig;
}

export interface DistributionAntiRollbackConfig {
  enabled?: boolean;
  requireExpectedHash?: boolean;
  requireExpectedSize?: boolean;
  requireManifestVersionSet?: boolean;
}

export interface DistributionConfigLike {
  sourceOrder?: DistributionSource[];
  sources?: DistributionSource[];
  sourceMatrix?: {
    cache?: { onHit?: 'return'; onMiss?: 'next' | 'terminal'; onFailure?: 'next' | 'terminal' };
    p2p?: { onHit?: 'return'; onMiss?: 'next' | 'terminal'; onFailure?: 'next' | 'terminal' };
    http?: { onHit?: 'return'; onMiss?: 'next' | 'terminal'; onFailure?: 'next' | 'terminal' };
  };
  p2p?: DistributionP2PConfig;
  sourceDecision?: DistributionSourceDecisionConfig;
  antiRollback?: DistributionAntiRollbackConfig;
  requiredContentEncoding?: string | null;
  expectedHash?: string | null;
  maxRetries?: number;
  initialRetryDelayMs?: number;
  maxRetryDelayMs?: number;
}

export interface DownloadShardOptions {
  sourceOrder?: DistributionSource[];
  distributionConfig?: DistributionConfigLike;
  distribution?: DistributionConfigLike;
  maxRetries?: number;
  initialRetryDelayMs?: number;
  maxRetryDelayMs?: number;
  requiredEncoding?: string | null;
  algorithm: string;
  signal?: AbortSignal;
  onProgress?: ((progress: DistributionProgress) => void) | null;
  onDeliveryMetrics?: ((event: {
    schemaVersion: number;
    shardIndex: number;
    source: DistributionSource | null;
    path: string | null;
    expectedManifestVersionSet: string | null;
    deliveryMetrics: ShardDeliveryMetrics;
    decisionTrace: ShardDeliveryDecisionTrace | null;
  }) => void | Promise<void>) | null;
  writeToStore?: boolean;
  enableSourceCache?: boolean;
  p2pTransport?: P2PTransport | null;
  expectedHash?: string | null;
  expectedSize?: number | null;
  expectedManifestVersionSet?: string | null;
}

export interface ShardDeliveryPlanStep {
  source: DistributionSource;
  enabled: boolean;
  reason: string;
}

export interface ShardDeliveryPlan {
  order: DistributionSource[];
  plan: ShardDeliveryPlanStep[];
}

export interface ShardDeliveryDecisionTrace {
  schemaVersion: number;
  deterministic: boolean;
  shardIndex: number;
  expectedManifestVersionSet: string | null;
  sourceOrder: DistributionSource[];
  plan: ShardDeliveryPlanStep[];
  attempts: Array<{
    source: DistributionSource;
    status: 'success' | 'failed' | 'skipped';
    reason: string | null;
    code: string | null;
    message: string | null;
    durationMs: number | null;
    bytes: number | null;
    hash: string | null;
    path: string | null;
    manifestVersionSet: string | null;
  }>;
}

export interface ShardDeliveryMetrics {
  schemaVersion: number;
  totalDurationMs: number;
  sourceOrder: DistributionSource[];
  successSource: DistributionSource | null;
  attemptCount: number;
  sourceAttempts: {
    cache: number;
    p2p: number;
    http: number;
  };
  retries: {
    cache: number;
    p2p: number;
    http: number;
  };
  failureCodes: Record<string, number>;
  p2pRttMs: {
    count: number;
    min: number | null;
    max: number | null;
    avg: number | null;
  };
  httpRttMs: {
    count: number;
    min: number | null;
    max: number | null;
    avg: number | null;
  };
  storageWriteMs: number | null;
}

export interface DownloadShardResult {
  buffer: ArrayBuffer | null;
  bytes: number;
  hash: string;
  wrote: boolean;
  source: DistributionSource;
  path: string;
  manifestVersionSet?: string | null;
  decisionTrace?: ShardDeliveryDecisionTrace;
  deliveryMetrics: ShardDeliveryMetrics;
}

export declare function resolveShardDeliveryPlan(options?: {
  sourceOrder?: DistributionSource[];
  enableSourceCache?: boolean;
  p2pEnabled?: boolean;
  p2pTransportAvailable?: boolean;
  httpEnabled?: boolean;
}): ShardDeliveryPlan;

export declare function downloadShard(
  baseUrl: string,
  shardIndex: number,
  shardInfo: DistributionShardInfo,
  options?: DownloadShardOptions
): Promise<DownloadShardResult>;

export declare function getSourceOrder(
  config?: DistributionConfigLike
): DistributionSource[];

export declare function getInFlightShardDeliveryCount(): number;
