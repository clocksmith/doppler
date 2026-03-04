export declare const P2P_OBSERVABILITY_SCHEMA_VERSION: 1;

export interface P2PDeliveryObservabilityRecord {
  schemaVersion: number;
  timestampMs: number;
  modelId: string | null;
  shardIndex: number | null;
  successSource: string | null;
  attemptCount: number;
  p2pAttempts: number;
  httpAttempts: number;
  cacheAttempts: number;
  totalDurationMs: number;
  p2pRttAvgMs: number | null;
  httpRttAvgMs: number | null;
  totalFailures: number;
  fallbackToHttp: boolean;
  p2pHit: boolean;
  failureCodes: Record<string, number>;
  rawMetrics: Record<string, unknown>;
}

export interface P2PObservabilitySummary {
  schemaVersion: number;
  generatedAtMs: number;
  totals: {
    records: number;
    successful: number;
    failed: number;
    p2pHits: number;
    httpFallbacks: number;
  };
  rates: {
    availability: number;
    p2pHitRate: number;
    httpFallbackRate: number;
  };
  latencyMs: {
    p50: number | null;
    p95: number | null;
    p99: number | null;
  };
  failureCodes: Record<string, number>;
  slo: {
    targets: {
      minAvailability: number;
      minP2PHitRate: number;
      maxHttpFallbackRate: number;
      maxP95LatencyMs: number;
    };
    breaches: Array<{
      id: string;
      metric: string;
      expected: string;
      actual: number;
    }>;
    status: 'pass' | 'fail';
  };
}

export interface P2PObservabilityAlert {
  schemaVersion: number;
  id: string;
  severity: 'warning' | 'critical';
  message: string;
  metric: string | null;
  expected: string | null;
  actual: number | null;
  generatedAtMs: number;
}

export declare function createP2PDeliveryObservabilityRecord(
  input: Record<string, unknown>,
  context?: {
    timestampMs?: number;
    modelId?: string | null;
    shardIndex?: number | null;
  }
): P2PDeliveryObservabilityRecord;

export declare function aggregateP2PDeliveryObservability(
  records?: Array<Record<string, unknown>>,
  options?: {
    targets?: {
      minAvailability?: number;
      minP2PHitRate?: number;
      maxHttpFallbackRate?: number;
      maxP95LatencyMs?: number;
    };
  }
): P2PObservabilitySummary;

export declare function buildP2PAlertsFromSummary(
  summary: P2PObservabilitySummary,
  options?: {
    escalateBreaches?: string[];
  }
): P2PObservabilityAlert[];

export declare function buildP2PDashboardSnapshot(
  records?: Array<Record<string, unknown>>,
  options?: {
    targets?: {
      minAvailability?: number;
      minP2PHitRate?: number;
      maxHttpFallbackRate?: number;
      maxP95LatencyMs?: number;
    };
    escalateBreaches?: string[];
  }
): {
  schemaVersion: number;
  generatedAtMs: number;
  summary: P2PObservabilitySummary;
  alerts: P2PObservabilityAlert[];
};
