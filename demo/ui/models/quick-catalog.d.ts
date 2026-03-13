export interface QuickCatalogEntryLike {
  modelId: string;
}

export declare function mergeQuickCatalogEntryLists<T extends QuickCatalogEntryLike>(
  entryLists: Array<T[] | null | undefined> | null | undefined
): T[];
