export type SelfSpeculationMode = 'none' | 'lookup' | 'self' | 'draft' | 'medusa';
export type SelfSpeculationVerifyMode = 'greedy';

export interface SelfSpeculationConfig {
  mode: SelfSpeculationMode;
  tokens: number;
  maxMatchTokens?: number;
  minMatchTokens?: number;
  maxRejectedBlocks?: number;
  verify: SelfSpeculationVerifyMode;
  threshold: number | null;
  rollbackOnReject: boolean;
}

export const SPECULATION_MODES: readonly SelfSpeculationMode[];
export const SPECULATION_VERIFY_MODES: readonly SelfSpeculationVerifyMode[];
export const DEFAULT_SELF_SPECULATION_CONFIG: Readonly<SelfSpeculationConfig>;

export function validateSelfSpeculationConfig(config: SelfSpeculationConfig): void;

export function resolveSelfSpeculationConfig(session: SelfSpeculationConfig | null | undefined, override: Partial<SelfSpeculationConfig> | null | undefined): Readonly<SelfSpeculationConfig> | null;
