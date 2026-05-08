export function isWeightsRefLane(entry) {
  return entry?.artifactCompleteness === 'weights-ref';
}

export function isPrimaryWeightPackLane(entry, weightPackId) {
  return entry?.weightPackId === weightPackId
    && entry?.artifactCompleteness === 'complete'
    && entry?.weightsRefAllowed === false;
}

export function isManifestOwnedLane(entry) {
  return entry?.runtimePromotionState === 'manifest-owned';
}

export function isVerifiedManifestOwnedLane(entry) {
  return isManifestOwnedLane(entry)
    && entry?.lifecycle?.status?.runtime === 'active'
    && entry?.lifecycle?.status?.tested === 'verified';
}

export function findPrimaryForWeightPack(catalogEntries, weightPackId) {
  if (!Array.isArray(catalogEntries) || !weightPackId) return null;
  for (const entry of catalogEntries) {
    if (isPrimaryWeightPackLane(entry, weightPackId)) return entry;
    if (isPrimaryWeightPackLane(entry?.demoFallbackVariant, weightPackId)) {
      return entry.demoFallbackVariant;
    }
  }
  return null;
}

export function findRegisteredSiblingsOf(primaryEntry, catalogEntries, storedModelIds) {
  if (!primaryEntry || primaryEntry.weightsRefAllowed !== false) return [];
  if (primaryEntry.artifactCompleteness !== 'complete') return [];
  const storedSet = storedModelIds instanceof Set
    ? storedModelIds
    : new Set(Array.isArray(storedModelIds) ? storedModelIds : []);
  return (Array.isArray(catalogEntries) ? catalogEntries : []).filter((entry) =>
    entry?.weightPackId === primaryEntry.weightPackId
    && entry?.artifactCompleteness === 'weights-ref'
    && entry?.weightsRefAllowed === true
    && storedSet.has(entry.modelId)
  );
}

function defaultIsVisibleEntry(entry) {
  if (entry?.demoVisible === false) return false;
  return entry?.quickstart === true || entry?.demoVisible === true;
}

function hasHfSource(entry) {
  const hf = entry?.hf;
  return typeof hf?.repoId === 'string'
    && hf.repoId.trim().length > 0
    && typeof hf?.revision === 'string'
    && hf.revision.trim().length > 0
    && typeof hf?.path === 'string'
    && hf.path.trim().length > 0;
}

function defaultHasSource(entry, localBaseUrls) {
  if (localBaseUrls?.has?.(entry?.modelId)) return true;
  if (typeof entry?.localBaseUrl === 'string' && entry.localBaseUrl.trim()) return true;
  if (typeof entry?.baseUrl === 'string' && entry.baseUrl.trim()) return true;
  return hasHfSource(entry);
}

function buildSurfacedWeightsRefEntry(weightsRefEntry, primary, localBaseUrls) {
  const merged = { ...weightsRefEntry };
  for (const key of ['demoWarningBadges', 'demoWarningText', 'recommended', 'sortOrder', 'sizeBytes', 'quickstart']) {
    if (merged[key] == null) {
      merged[key] = primary[key];
    }
  }
  merged.demoVisible = true;
  merged.localBaseUrl = localBaseUrls?.get?.(weightsRefEntry.modelId) ?? null;
  merged.weightsRefPrimary = primary.modelId;
  merged.demoFallbackVariant = {
    ...primary,
    localBaseUrl: localBaseUrls?.get?.(primary.modelId) ?? null,
    weightsRefPrimary: null,
    demoFallbackVariant: null,
  };
  merged.laneSelection = {
    kind: 'preferred_weights_ref',
    visibleModelId: primary.modelId,
    preferredModelId: weightsRefEntry.modelId,
    fallbackModelId: primary.modelId,
    weightPackId: weightsRefEntry.weightPackId ?? primary.weightPackId ?? null,
  };
  return merged;
}

export function selectCatalogModelLanes(catalogEntries, options = {}) {
  const entries = Array.isArray(catalogEntries) ? catalogEntries : [];
  const localBaseUrls = options.localBaseUrls instanceof Map ? options.localBaseUrls : new Map();
  const isVisibleEntry = typeof options.isVisibleEntry === 'function'
    ? options.isVisibleEntry
    : defaultIsVisibleEntry;
  const hasSource = typeof options.hasSource === 'function'
    ? options.hasSource
    : (entry) => defaultHasSource(entry, localBaseUrls);

  const primaryByWeightPackId = new Map();
  for (const entry of entries) {
    if (!entry?.modes?.includes('text')) continue;
    if (!isVisibleEntry(entry)) continue;
    if (!isManifestOwnedLane(entry)) continue;
    if (!isPrimaryWeightPackLane(entry, entry?.weightPackId)) continue;
    if (!hasSource(entry)) continue;
    if (entry?.weightPackId) {
      primaryByWeightPackId.set(entry.weightPackId, entry);
    }
  }

  const preferredByPrimaryId = new Map();
  for (const entry of entries) {
    if (!entry?.modes?.includes('text')) continue;
    if (!isWeightsRefLane(entry)) continue;
    if (entry?.weightsRefAllowed !== true) continue;
    if (!isVerifiedManifestOwnedLane(entry)) continue;
    if (!hasSource(entry)) continue;
    const primary = primaryByWeightPackId.get(entry.weightPackId);
    if (!primary) continue;
    const requestedId = typeof primary.demoPreferredVariantId === 'string'
      ? primary.demoPreferredVariantId.trim()
      : '';
    if (!requestedId || requestedId !== entry.modelId) continue;
    preferredByPrimaryId.set(primary.modelId, entry);
  }

  const selected = [];
  for (const entry of entries) {
    if (!entry?.modes?.includes('text')) continue;
    if (!isVisibleEntry(entry)) continue;
    if (!isManifestOwnedLane(entry)) continue;
    if (!isPrimaryWeightPackLane(entry, entry?.weightPackId)) continue;
    if (!hasSource(entry)) continue;

    const preferred = preferredByPrimaryId.get(entry.modelId);
    if (preferred) {
      selected.push(buildSurfacedWeightsRefEntry(preferred, entry, localBaseUrls));
    } else {
      selected.push({
        ...entry,
        localBaseUrl: localBaseUrls.get(entry.modelId) ?? null,
        demoFallbackVariant: null,
        weightsRefPrimary: null,
        laneSelection: {
          kind: 'primary',
          visibleModelId: entry.modelId,
          preferredModelId: entry.modelId,
          fallbackModelId: null,
          weightPackId: entry.weightPackId ?? null,
        },
      });
    }
  }

  selected.sort((a, b) => {
    if (a.recommended !== b.recommended) {
      return a.recommended ? -1 : 1;
    }
    if ((a.sortOrder ?? 999) !== (b.sortOrder ?? 999)) {
      return (a.sortOrder ?? 999) - (b.sortOrder ?? 999);
    }
    return String(a.label || a.modelId || '').localeCompare(String(b.label || b.modelId || ''));
  });
  return selected;
}
