export interface DistillDataScope {
  sourceLangs: string[] | null;
  targetLangs: string[] | null;
  pairAllowlist: string[] | null;
  sourceLangSet: Set<string> | null;
  targetLangSet: Set<string> | null;
  pairAllowlistSet: Set<string> | null;
  strictPairContract: boolean;
}

export interface DistillSample {
  index?: number;
  direction?: string | null;
  sourceLang?: string | null;
  targetLang?: string | null;
  source?: string | null;
  targetPos?: string | null;
  targetNeg?: string | null;
}

export declare function normalizeOptionalString(value: unknown): string | null;

export declare function normalizeDistillDatasetPath(value: unknown): string | null;

export declare function resolveDistillDataScope(
  options?: Record<string, unknown>,
  trainingConfig?: Record<string, unknown> | null
): DistillDataScope;

export declare function encodeDistillRow(
  record: Record<string, unknown> | null | undefined,
  index: number,
  scope?: DistillDataScope | null
): DistillSample | null;

export declare function summarizeDirectionCounts(
  samples: Array<Record<string, unknown> | null | undefined>
): Record<string, number>;

export declare function buildDistillPrompt(sample: Record<string, unknown> | null | undefined): string;

export declare function buildDistillCandidatePrompt(
  sample: Record<string, unknown> | null | undefined,
  candidate: unknown
): string;
