export declare const DEFAULT_FINITENESS_ABS_THRESHOLD: number;
export declare const DEFAULT_DEFERRED_ROUNDING_WINDOW_TOKENS: number;

export interface RangeAwareSelectiveWideningConfig {
  enabled: boolean;
  includeNonFinite: boolean;
  absThreshold: number;
}

export declare function resolveRangeAwareSelectiveWideningConfig(
  computeConfig: {
    rangeAwareSelectiveWidening?: {
      enabled?: boolean;
      includeNonFinite?: boolean;
      absThreshold?: number;
    };
  } | null | undefined
): RangeAwareSelectiveWideningConfig;

export declare function resolveDeferredRoundingWindowTokens(
  computeConfig: {
    deferredRoundingWindowTokens?: number;
  } | null | undefined
): number;

export declare function shouldRunFinitenessGuard(
  activationDtype: 'f16' | 'f32' | string | undefined,
  computeConfig: {
    rangeAwareSelectiveWidening?: {
      enabled?: boolean;
      includeNonFinite?: boolean;
      absThreshold?: number;
    };
  } | null | undefined
): boolean;
