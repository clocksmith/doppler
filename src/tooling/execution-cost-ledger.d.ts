export declare const TOKEN_COST_LEDGER_SCHEMA: 'doppler.token-cost-ledger/v1';

export declare function isExecutionObservationRequested(
  runtimeConfig: Record<string, unknown>
): boolean;

export declare function buildTokenCostLedger(options: {
  metrics: Record<string, unknown>;
  identity?: Record<string, unknown>;
  device?: Record<string, unknown> | null;
  browser?: Record<string, unknown> | null;
}): Record<string, unknown>;

export declare function classifyTokenCostLedger(
  ledger: Record<string, unknown>,
  policy: Record<string, unknown>
): {
  dominantWall: string;
  classifiedGpuMs: Array<{ wall: string; gpuMs: number }>;
  prescribedExperiments: string[];
};
