/**
 * Hot-Swap Config Schema
 *
 * Security policy for swapping JS/WGSL/JSON artifacts at runtime.
 *
 * @module config/schema/hotswap
 */

// =============================================================================
// Signer Schema
// =============================================================================

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

// =============================================================================
// Hot-Swap Config
// =============================================================================

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
  /** Allowlisted signers for distributed bundles */
  trustedSigners: HotSwapSignerSchema[];
  /** Optional manifest URL for test harness workflows */
  manifestUrl: string | null;
}

/** Default hot-swap configuration */
export const DEFAULT_HOTSWAP_CONFIG: HotSwapConfigSchema = {
  enabled: false,
  localOnly: false,
  allowUnsignedLocal: false,
  trustedSigners: [],
  manifestUrl: null,
};
