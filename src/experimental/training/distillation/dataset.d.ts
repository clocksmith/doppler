export interface CanonicalTranslationRow {
  row_id: string;
  src_lang: string | null;
  tgt_lang: string | null;
  pair: string | null;
  source: string;
  target_pos: string;
  target_neg: string | null;
}

export interface CanonicalTranslationDataset {
  absolutePath: string;
  raw: string;
  rows: CanonicalTranslationRow[];
  rowCount: number;
  directionCounts: Record<string, number>;
  datasetHash: string;
  canonicalHash: string;
  rowIdsHash: string;
}

export declare function normalizeDistillationPair(
  value: unknown,
  srcLang?: string | null,
  tgtLang?: string | null
): string | null;

export declare function normalizeTranslationPairRow(
  record: Record<string, unknown>,
  index: number,
  options?: { strictPairContract?: boolean }
): CanonicalTranslationRow | null;

export declare function loadCanonicalTranslationDataset(
  datasetPath: string,
  options?: {
    strictPairContract?: boolean;
    sourceLangs?: string[] | string | null;
    targetLangs?: string[] | string | null;
    pairAllowlist?: string[] | string | null;
  }
): Promise<CanonicalTranslationDataset>;

export declare function buildFrozenSubset(options: {
  datasetPath: string;
  outputDir: string;
  strictPairContract?: boolean;
  sourceLangs?: string[] | string | null;
  targetLangs?: string[] | string | null;
  pairAllowlist?: string[] | string | null;
  subsetSpec?: Record<string, unknown> | null;
}): Promise<{
  dataset: CanonicalTranslationDataset;
  subsetRows: CanonicalTranslationRow[];
  subsetJsonlPath: string;
  rowIdsPath: string;
  manifestPath: string;
  manifest: Record<string, unknown>;
}>;
