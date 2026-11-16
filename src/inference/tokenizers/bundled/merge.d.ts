/** Merge ranks are the unique ordinal positions produced by buildMergeRanks. */
export function mergeBpeTokens(text: string, ranks: Pick<ReadonlyMap<string, number>, 'get'>): string[];
