import type { RuntimeSamplingOptions } from '../../../config/generation-contract.js';

export interface ResolvedSamplingConfig extends RuntimeSamplingOptions {
  greedyThreshold: number;
  suppressSpecialTokens: boolean;
  suppressSpecialLikeTokens: boolean;
}

export type SamplingCallOptions = Partial<Omit<ResolvedSamplingConfig, 'suppressTokenIds'>> & {
  suppressTokenIds?: readonly number[];
};

export function resolveSamplingConfig(
  opts: SamplingCallOptions | null | undefined,
  runtimeConfig: { inference?: { sampling?: ResolvedSamplingConfig } } | null | undefined
): ResolvedSamplingConfig;
