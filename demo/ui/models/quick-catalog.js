export function mergeQuickCatalogEntryLists(entryLists) {
  const merged = [];
  const seenModelIds = new Set();
  for (const entries of entryLists || []) {
    if (!Array.isArray(entries)) continue;
    for (const entry of entries) {
      const modelId = typeof entry?.modelId === 'string' ? entry.modelId.trim() : '';
      if (!modelId || seenModelIds.has(modelId)) continue;
      seenModelIds.add(modelId);
      merged.push(entry);
    }
  }
  return merged;
}
