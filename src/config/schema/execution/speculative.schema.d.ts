/**
 * Speculative decoding configuration.
 *
 * @module config/schema/speculative
 */

export interface SpeculativeConfigSchema {
  /** Draft token count per speculative step */
  numDraftTokens: number;
  /** Max rejection retries before falling back */
  maxRejectionRetries: number;
  /** Enable tree-based draft sampling */
  enableTreeDraft: boolean;
  /** Temperature for speculative sampling */
  temperature: number;
  /** Optional deterministic seed for speculative sampling */
  randomSeed: number | null;
}

/** Default speculative decoding configuration */
export declare const DEFAULT_SPECULATIVE_CONFIG: SpeculativeConfigSchema;
