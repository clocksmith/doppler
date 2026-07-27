export declare const TOKEN_COST_LEDGER_SCHEMA: 'doppler.token-cost-ledger/v1';

export interface TokenCostHostCosts {
  recordMs: number | null;
  submitWaitMs: number | null;
  readbackWaitMs: number | null;
  fenceWaitMs: number | null;
  orchestrationMs: number | null;
  readbackBreakdown: {
    mapWaitMs: number | null;
    cleanupMs: number | null;
    copyMs: number | null;
  };
  observedSerialMs: number | null;
  dominantWall:
    | 'command-recording'
    | 'submit-readback-fence'
    | 'host-orchestration'
    | null;
  overlapSemantics: string;
}

export interface TokenCostOperation {
  label: string;
  gpuMs: number | null;
  dispatches: number;
  knownDispatchGeometry: number;
  workgroups: number;
}

export interface TokenCostPhase {
  phase: 'prefill' | 'decode';
  measurementSource: 'gpu-timestamp-query' | 'cpu-wall-estimate';
  wallMs: number | null;
  attributedGpuMs: number | null;
  unattributedWallMs: number | null;
  timestampCoverageRatio: number | null;
  overlapSemantics: string;
  hostCosts: TokenCostHostCosts;
  operations: TokenCostOperation[];
  dispatches: number;
  unattributedDispatches: number;
  dispatchGeometryCoverage: number | null;
  commandBufferSubmissions: number;
  observerReadbacks: number;
  executionReadbacks: { value: null; status: string };
  estimatedBytesMoved: {
    value: null;
    unit: 'bytes';
    status: 'unavailable';
    semantics: 'estimated-not-measured';
  };
}

export interface TokenCostLedger {
  schema: 'doppler.token-cost-ledger/v1';
  identity: Record<string, string | null>;
  device: Record<string, unknown> | null;
  browser: Record<string, unknown> | null;
  phases: TokenCostPhase[];
  selectedVariants: Record<string, unknown>;
  rejectedOrFallbackVariants: unknown[];
  dominantOperation: {
    phase: string;
    label: string;
    gpuMs: number;
  } | null;
  dominantObservedWall: {
    phase: string;
    wall: string;
    ms: number;
  } | null;
  digest: `sha256:${string}`;
}

export declare function isExecutionObservationRequested(
  runtimeConfig: Record<string, unknown>
): boolean;

export declare function buildTokenCostLedger(options: {
  metrics: Record<string, unknown>;
  identity?: Record<string, unknown>;
  device?: Record<string, unknown> | null;
  browser?: Record<string, unknown> | null;
}): TokenCostLedger;

export declare function classifyTokenCostLedger(
  ledger: TokenCostLedger,
  policy: Record<string, unknown>
): {
  dominantWall: string;
  classifiedGpuMs: Array<{ wall: string; gpuMs: number }>;
  classifiedHostMs: Array<{ wall: string; ms: number }>;
  prescribedExperiments: string[];
};
