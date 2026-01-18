/**
 * Intent Bundle Config Schema
 *
 * Runtime policy for intent bundle gating.
 *
 * @module config/schema/intent-bundle
 */

/**
 * Intent bundle configuration.
 */
export interface IntentBundleConfigSchema {
  /** Enable intent bundle checks */
  enabled: boolean;
  /** Optional URL for fetching bundle in test harness workflows */
  bundleUrl: string | null;
  /** Loaded intent bundle (runtime-only) */
  bundle: Record<string, unknown> | null;
  /** Require base model hash to match the loaded manifest */
  requireBaseModelHash: boolean;
  /** Require kernel registry version match */
  requireKernelRegistryVersion: boolean;
  /** Require deterministic output hash if specified in bundle */
  enforceDeterministicOutput: boolean;
}

/** Default intent bundle configuration */
export declare const DEFAULT_INTENT_BUNDLE_CONFIG: IntentBundleConfigSchema;
