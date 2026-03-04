/**
 * Hot-Swap Config Schema
 *
 * Security policy for swapping JS/WGSL/JSON artifacts at runtime.
 *
 * @module config/schema/hotswap
 */

/**
 * Trusted signer entry.
 *
 * `publicKeyJwk` is used for signature verification.
 */
export interface HotSwapSignerSchema {
  /** Stable signer ID */
  id: string;
  /** Public key in JWK format */
  publicKeyJwk: JsonWebKey;
}

export type HotSwapRolloutMode = 'shadow' | 'canary' | 'opt-in' | 'default';

export interface HotSwapRolloutSchema {
  /** Rollout mode for production orchestration */
  mode: HotSwapRolloutMode;
  /** Canary percentage (0-100) for mode="canary" */
  canaryPercent: number;
  /** Deterministic cohort salt for bucket selection */
  cohortSalt: string;
  /** Optional allowlist for mode="opt-in" */
  optInAllowlist: string[];
}

/**
 * Hot-swap configuration.
 */
export interface HotSwapConfigSchema {
  /** Enable hot-swap loading (default: false) */
  enabled: boolean;
  /** Treat swaps as local-only (no distribution) */
  localOnly: boolean;
  /** Allow unsigned bundles when localOnly is true */
  allowUnsignedLocal: boolean;
  /** Policy schema version */
  policyVersion: number;
  /** Rollout policy for manifest activation */
  rollout: HotSwapRolloutSchema;
  /** Allowlisted signers for distributed bundles */
  trustedSigners: HotSwapSignerSchema[];
  /** Optional manifest URL for test harness workflows */
  manifestUrl: string | null;
}

/** Default hot-swap configuration */
export declare const DEFAULT_HOTSWAP_CONFIG: HotSwapConfigSchema;
