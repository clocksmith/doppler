export interface ExecutionPlanSummary {
  id: string;
  kernelPathId: string | null;
  kernelPathSource: string;
  activationDtype: string;
  readbackInterval: number | null;
  readbackMode: string | null;
  batchSize: number | undefined;
  stopCheckMode: string | undefined;
  ringTokens: number | null;
  ringStop: number | null;
  ringStaging: number | null;
}

export declare function shouldDisableBatchDecodeAfterShortBatch(input: {
  hitStop?: boolean;
  actualCount?: number;
  requestedCount?: number;
}): boolean;

export declare function resolveHotVocabularyBatchDecodeAvailability(input: {
  hasRangeBackedPerLayerInputs?: boolean;
  pleHotVocabularyRuntime?: Record<string, unknown> | null;
  tokenId?: number;
}): boolean;

export declare function primePleDecodeRuntimeCache(
  state: Record<string, unknown>,
  seedTokenIds?: number[] | null
): Promise<void>;

export declare function recordPrefillProfileStep(
  state: Record<string, unknown>,
  entry: Record<string, unknown>
): void;

/**
 * Resolve display text for token IDs. Custom primary and fallback renderers
 * are tried around tokenizer.decode, preserving empty-string special-token
 * filtering when fallback text looks like a special token.
 */
export declare function resolveTokenText(
  tokenizer: { decode?(ids: readonly number[], skipSpecialTokens?: boolean): string },
  tokenIds: readonly number[],
  fallbackText?: string,
  renderTokenText?: ((ids: readonly number[]) => string) | null,
  renderFallbackTokenText?: ((ids: readonly number[]) => string) | null
): string;

export declare function usesReplayPrefillDecode(state: Record<string, unknown> | null | undefined): boolean;

export declare function assertIncrementalDecodeSupport(
  state: Record<string, unknown>,
  operation: string
): void;

export declare function summarizeExecutionPlan(plan: Record<string, unknown> | null | undefined): ExecutionPlanSummary | null;

export declare function shouldRetryWithFinitenessFallback(error: unknown): boolean;

export declare function createUnhandledFinitenessPolicyError(
  state: Record<string, unknown>,
  contextLabel: string,
  error: unknown
): Error;

export declare function resolveTargetPlanKVDtype(
  plan: Record<string, unknown> | null | undefined,
  contextLabel: string
): 'f16' | 'f32';

export declare function resolveCurrentKVCacheDtype(
  state: Record<string, unknown>,
  plan: Record<string, unknown> | null | undefined,
  contextLabel: string
): 'f16' | 'f32';

export declare function cloneRuntimeInferenceWithKVDtype(
  state: Record<string, unknown>,
  kvDtype: 'f16' | 'f32'
): Record<string, unknown>;
