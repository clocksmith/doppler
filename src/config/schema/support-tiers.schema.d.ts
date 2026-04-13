/**
 * Support-tier registry schema.
 *
 * Canonical source for subsystem support classification used to generate
 * `docs/subsystem-support-matrix.md`.
 *
 * @module config/schema/support-tiers
 */

export type SupportTier = 'tier1' | 'experimental' | 'internal-only';
export type SupportScope =
  | 'api'
  | 'cli'
  | 'demo'
  | 'format'
  | 'runtime'
  | 'integration'
  | 'browser';
export type ClaimVisibility = 'primary' | 'secondary' | 'none';

export interface SupportSubsystemEntrySchema {
  id: string;
  label: string;
  scope: SupportScope;
  tier: SupportTier;
  owner: string;
  userFacing: boolean;
  demoDefault: boolean;
  exported: boolean;
  claimVisibility: ClaimVisibility;
  packageExport: string | null;
  command: string | null;
  docs: string[];
  entrypoints: string[];
  notes: string;
}

export interface SupportTierRegistrySchema {
  schemaVersion: number;
  source: 'doppler';
  updatedAtUtc: string;
  subsystems: SupportSubsystemEntrySchema[];
}

export declare const SUPPORT_TIER_REGISTRY_SCHEMA_VERSION: 1;
export declare const SUPPORT_TIERS: readonly SupportTier[];
export declare const SUPPORT_SCOPES: readonly SupportScope[];
export declare const CLAIM_VISIBILITY_LEVELS: readonly ClaimVisibility[];
export declare const DEFAULT_SUPPORT_TIER_REGISTRY: SupportTierRegistrySchema;

export declare function validateSupportTierRegistry(
  registry: unknown
): SupportTierRegistrySchema;
