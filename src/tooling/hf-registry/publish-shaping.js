import { ensureCatalogPayload, isPlainObject, normalizeText } from './catalog-io.js';

import { getEntryHfSpec } from './registry-urls.js';

function sortCatalogEntries(models) {
  models.sort((left, right) => {
    const leftSort = Number.isFinite(Number(left?.sortOrder)) ? Number(left.sortOrder) : Number.MAX_SAFE_INTEGER;
    const rightSort = Number.isFinite(Number(right?.sortOrder)) ? Number(right.sortOrder) : Number.MAX_SAFE_INTEGER;
    if (leftSort !== rightSort) {
      return leftSort - rightSort;
    }
    return normalizeText(left?.label || left?.modelId).localeCompare(
      normalizeText(right?.label || right?.modelId)
    );
  });
  return models;
}

export function isHostedRegistryApprovedEntry(entry) {
  return entry?.lifecycle?.availability?.hf === true
    && normalizeText(entry?.lifecycle?.status?.runtime) === 'active'
    && normalizeText(entry?.lifecycle?.status?.tested) === 'verified';
}


export function buildPublishedRegistryEntry(localEntry, revision) {
  const modelId = normalizeText(localEntry?.modelId);
  if (!modelId) {
    throw new Error('Published registry entry requires a non-empty modelId.');
  }
  const next = structuredClone(localEntry);
  const hf = isPlainObject(next.hf) ? next.hf : {};
  const hfSpec = getEntryHfSpec(next);
  const repoId = hfSpec.repoId;
  const repoPath = hfSpec.path;
  if (!repoId) {
    throw new Error(
      `Published registry entry for "${modelId}" requires explicit hf.repoId.`
    );
  }
  if (!repoPath) {
    throw new Error(
      `Published registry entry for "${modelId}" requires explicit hf.path.`
    );
  }
  const lifecycle = isPlainObject(next.lifecycle) ? next.lifecycle : {};
  const availability = isPlainObject(lifecycle.availability) ? lifecycle.availability : {};
  next.hf = {
    ...hf,
    repoId,
    revision: normalizeText(revision),
    path: repoPath,
  };
  next.lifecycle = {
    ...lifecycle,
    availability: {
      ...availability,
      hf: true,
    },
  };
  return next;
}

export function buildHostedRegistryPayload(payload, revisionOverrides = new Map()) {
  const source = ensureCatalogPayload(payload, 'support registry');
  const normalizedOverrides = revisionOverrides instanceof Map ? revisionOverrides : new Map();
  const approved = Array.isArray(source.models)
    ? source.models.filter((entry) => isHostedRegistryApprovedEntry(entry))
    : [];
  const primaryWeightPackIds = collectPrimaryWeightPackIds(approved);
  const models = approved.map((entry) => {
    const shapeErrors = validateRegistryEntryArtifactIdentity(
      entry,
      'for hosted registry entries',
      { primaryWeightPackIds }
    );
    if (shapeErrors.length > 0) {
      throw new Error(shapeErrors.join('\n'));
    }
    const modelId = normalizeText(entry?.modelId);
    const revisionOverride = normalizeText(normalizedOverrides.get(modelId));
    if (revisionOverride) {
      return buildPublishedRegistryEntry(entry, revisionOverride);
    }
    return structuredClone(entry);
  });
  sortCatalogEntries(models);
  return {
    version: Number.isFinite(Number(source.version)) ? Number(source.version) : 1,
    lifecycleSchemaVersion: Number.isFinite(Number(source.lifecycleSchemaVersion))
      ? Number(source.lifecycleSchemaVersion)
      : 1,
    updatedAt: normalizeText(source.updatedAt) || new Date().toISOString().slice(0, 10),
    models,
  };
}

function validateRegistryEntryArtifactIdentity(entry, suffix, options = {}) {
  const errors = [];
  const modelId = normalizeText(entry?.modelId) || 'unknown-model';
  for (const field of ['sourceCheckpointId', 'weightPackId', 'manifestVariantId']) {
    if (!normalizeText(entry?.[field])) {
      errors.push(`${modelId}: ${field} is required ${suffix}`);
    }
  }
  // Two valid shapes:
  //   1. Primary lane: artifactCompleteness=complete, weightsRefAllowed=false.
  //      Self-contained — shards published next to manifest.
  //   2. Manifest-only sibling: artifactCompleteness=weights-ref,
  //      weightsRefAllowed=true. Must point at a primary lane published in
  //      the same payload.
  const completeness = entry?.artifactCompleteness;
  const weightsRefAllowed = entry?.weightsRefAllowed;
  if (typeof weightsRefAllowed !== 'boolean') {
    errors.push(`${modelId}: weightsRefAllowed must be a boolean ${suffix}`);
  }
  if (completeness === 'complete') {
    if (weightsRefAllowed === true) {
      errors.push(
        `${modelId}: artifactCompleteness="complete" requires weightsRefAllowed=false ${suffix}`
      );
    }
  } else if (completeness === 'weights-ref') {
    if (weightsRefAllowed !== true) {
      errors.push(
        `${modelId}: artifactCompleteness="weights-ref" requires weightsRefAllowed=true ${suffix}`
      );
    }
    const primaryWeightPackIds = options.primaryWeightPackIds;
    if (primaryWeightPackIds instanceof Set) {
      const weightPackId = normalizeText(entry?.weightPackId);
      if (weightPackId && !primaryWeightPackIds.has(weightPackId)) {
        errors.push(
          `${modelId}: artifactCompleteness="weights-ref" requires a primary lane ` +
          `with weightPackId="${weightPackId}" published in the same payload ${suffix}`
        );
      }
    }
  } else {
    errors.push(
      `${modelId}: artifactCompleteness must be "complete" or "weights-ref" ${suffix}`
    );
  }
  if (entry?.runtimePromotionState !== 'manifest-owned') {
    errors.push(`${modelId}: runtimePromotionState must be "manifest-owned" ${suffix}`);
  }
  return errors;
}

function collectPrimaryWeightPackIds(entries) {
  const ids = new Set();
  if (!Array.isArray(entries)) return ids;
  for (const entry of entries) {
    if (entry?.artifactCompleteness === 'complete' && entry?.weightsRefAllowed === false) {
      const weightPackId = normalizeText(entry?.weightPackId);
      if (weightPackId) {
        ids.add(weightPackId);
      }
    }
  }
  return ids;
}

export function validateLocalHfEntryShape(entry) {
  const errors = [];
  const modelId = normalizeText(entry?.modelId) || 'unknown-model';
  const hfSpec = getEntryHfSpec(entry);
  if (!hfSpec.repoId) {
    errors.push(`${modelId}: hf.repoId is required when lifecycle.availability.hf=true`);
  }
  if (!hfSpec.revision) {
    errors.push(`${modelId}: hf.revision is required when lifecycle.availability.hf=true`);
  }
  if (!hfSpec.path) {
    errors.push(`${modelId}: hf.path is required when lifecycle.availability.hf=true`);
  }
  errors.push(...validateRegistryEntryArtifactIdentity(entry, 'when lifecycle.availability.hf=true'));
  return errors;
}
