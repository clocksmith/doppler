export type DopplerClaimType =
  | 'execution-identity'
  | 'coarse-performance'
  | 'quality-inspection'
  | 'diagnostic';

export interface DopplerObservationPolicy {
  id: string;
  label: string;
  modifiesExecution: boolean;
  performanceRepresentative: boolean;
  requiredCaptures: string[];
  allowedClaimTypes: DopplerClaimType[];
  gpuTimestampQueries: boolean;
  perplexity: null | {
    wordSegmentation: 'doppler.word-segmentation/unicode-whitespace-v1';
    aggregation: 'doppler.perplexity/summed-word-surprisal-v1';
    rollingWindow: {
      unit: 'words' | 'tokens';
      size: number;
    };
  };
}

export interface DopplerComparisonFingerprint {
  schema: 'doppler.comparison-fingerprint/v1';
  identity: {
    artifact: {
      modelId: string;
      manifestHash: string;
    };
    tokenizer: {
      contract: unknown;
      digest: string;
    };
    promptTokenIds: number[];
    sampling: Record<string, unknown>;
    observationPolicy: {
      id: string;
      modifiesExecution: boolean;
      performanceRepresentative: boolean;
      requiredCaptures: string[];
      allowedClaimTypes: DopplerClaimType[];
    };
    perplexity: DopplerObservationPolicy['perplexity'];
    execution: Record<string, unknown>;
    browser: Record<string, string>;
    adapter: Record<string, string | null>;
  };
  fullDigest: string;
  qualityDigest: string;
  performanceDigest: string;
}

export interface DopplerInspectionTokenRecord {
  index: number;
  tokenId: number;
  text: string;
  probability: number | null;
  surprisal: number | null;
  topCandidates: Array<{
    tokenId: number;
    logit: number;
    text: string;
    probability: number | null;
  }>;
}

export interface DopplerWordQualityRecord {
  wordIndex: number;
  text: string;
  tokenIndexes: number[];
  tokenCount: number;
  summedSurprisal: number;
  probabilityAvailable: boolean;
  rollingPerplexity: number | null;
  cumulativePerplexity: number | null;
  rollingWindow: {
    unit: 'words' | 'tokens';
    size: number;
    tokenCount: number;
  };
}

export declare const OBSERVATION_POLICY_REGISTRY_SCHEMA: 'doppler.observation-policy-registry/v1';
export declare const COMPARISON_FINGERPRINT_SCHEMA: 'doppler.comparison-fingerprint/v1';
export declare const MODEL_INSPECTION_RECEIPT_SCHEMA: 'doppler.model-inspection-receipt/v1';
export declare const WORD_SEGMENTATION_SCHEMA: 'doppler.word-segmentation/unicode-whitespace-v1';
export declare const PERPLEXITY_AGGREGATION_SCHEMA: 'doppler.perplexity/summed-word-surprisal-v1';

export declare function listObservationPolicies(): DopplerObservationPolicy[];
export declare function resolveObservationPolicy(policyId?: string): DopplerObservationPolicy;
export declare function buildComparisonFingerprint(input: {
  artifact: { modelId: string; manifestHash: string };
  tokenizer: unknown;
  promptTokenIds: readonly number[];
  sampling?: Record<string, unknown>;
  observationPolicyId: string;
  execution?: Record<string, unknown>;
  browser?: Record<string, unknown>;
  adapter?: Record<string, unknown>;
}): DopplerComparisonFingerprint;
export declare function assertComparableFingerprints(
  kind: 'quality' | 'performance',
  left: DopplerComparisonFingerprint,
  right: DopplerComparisonFingerprint
): true;
export declare function buildInspectionTokenRecords(
  tokenIds: readonly number[],
  logitsByStep: ArrayLike<number>[],
  tokenizer: { decode(ids: readonly number[], skipSpecialTokens?: boolean, cleanUp?: boolean): string },
  topKSize?: number
): DopplerInspectionTokenRecord[];
export declare function aggregateWordPerplexity(
  tokenRecords: DopplerInspectionTokenRecord[],
  options?: { windowUnit?: 'words' | 'tokens'; windowSize?: number }
): {
  wordSegmentation: 'doppler.word-segmentation/unicode-whitespace-v1';
  aggregation: 'doppler.perplexity/summed-word-surprisal-v1';
  rollingWindow: { unit: 'words' | 'tokens'; size: number };
  words: DopplerWordQualityRecord[];
};
