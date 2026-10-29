import type { SampleStats } from '../../debug/stats.js';
export interface RequestTimingSample {
  scope: 'generation-call-through-completion';
  firstTokenMs: number | null;
  totalRunMs: number;
  decodeMs: number | null;
  decodeTokens: number;
  tokensGenerated: number;
  decodeTokensPerSec: number;
  perTokenMs: number[];
}
export declare function buildRequestTiming(start: number, durationMs: number, tokenTimestamps: number[]): RequestTimingSample;
export declare function summarizeRequestTiming(samples: RequestTimingSample[], prefillTokens: number[], timedRuns: number, modelLoadMs: number, samplePolicy: { outlierPolicy: 'none' | 'iqr'; outlierIqrMultiplier?: number }): {
  metrics: { scope: string; samples: RequestTimingSample[]; latency: Record<string, SampleStats>; perTokenMs: SampleStats };
  timing: Record<string, number>;
  scope: string;
};
