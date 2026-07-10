export type SelfSpeculationMode = 'none' | 'self' | 'draft' | 'medusa';
export type SelfSpeculationVerifyMode = 'greedy';

export interface SelfSpeculationConfig {
  mode: SelfSpeculationMode;
  tokens: number;
  verify: SelfSpeculationVerifyMode;
  threshold: number | null;
  rollbackOnReject: boolean;
}

export const SPECULATION_MODES: readonly SelfSpeculationMode[];
export const SPECULATION_VERIFY_MODES: readonly SelfSpeculationVerifyMode[];
export const DEFAULT_SELF_SPECULATION_CONFIG: Readonly<SelfSpeculationConfig>;

export function validateSelfSpeculationConfig(config: SelfSpeculationConfig): void;
