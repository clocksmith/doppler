import type { InspectedToken } from '../ui/token-inspector/index.js';

export declare const SAMPLE_INSPECTION_RECEIPT: {
  policy: { id: string; label: string; modifiesExecution: boolean; performanceRepresentative: boolean };
  prompt: string;
  outputText: string;
  generatedTokenIds: number[];
  wallTimingMs: number;
  generationEvidence: { stats: { prefillTimeMs: number; decodeTimeMs: number; tokensPerSecond: number } };
  fingerprint: {
    identity: {
      execution: { backend: string; precision: string; pipeline: string };
      adapter: { vendor: string; architecture: string };
    };
    fullDigest: string;
    qualityDigest: string;
    performanceDigest: string;
  };
  quality: {
    words: Array<{
      text: string;
      rollingPerplexity: number;
      summedSurprisal: number;
      cumulativePerplexity: number;
      tokenCount: number;
      rollingWindow: { size: number; unit: string };
    }>;
  };
  tokens: Array<InspectedToken & { index: number }>;
};
