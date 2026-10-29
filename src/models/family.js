// Shared factory for the per-family pointer modules (qwen3, gemma3, gemma4,
// embeddinggemma, diffusiongemma). Entries derive from models/catalog.json,
// the canonical model catalog; family modules hold no model data of their own.

import catalog from '../../models/catalog.json' with { type: 'json' };

function toFamilyEntry(model) {
  return Object.freeze({
    modelId: model.modelId,
    label: model.label,
    sourceModel: model.sourceCheckpointId,
    hfRepoId: model.hf?.repoId ?? null,
    hfPath: model.hf?.path ?? null,
    modes: Object.freeze([...model.modes]),
  });
}

export function createFamily({ familyId }) {
  const KNOWN_MODELS = Object.freeze(
    catalog.models.filter((model) => model.family === familyId).map(toFamilyEntry)
  );
  if (KNOWN_MODELS.length === 0) {
    throw new Error(`models/catalog.json has no models for family "${familyId}".`);
  }
  const repoIds = [...new Set(KNOWN_MODELS.map((entry) => entry.hfRepoId).filter(Boolean))];

  function resolveModel(modelId) {
    return KNOWN_MODELS.find((m) => m.modelId === modelId) || null;
  }

  function resolveHfBaseUrl(modelId, revision = 'main') {
    const entry = resolveModel(modelId);
    if (!entry?.hfRepoId || !entry.hfPath) return null;
    return `https://huggingface.co/${entry.hfRepoId}/resolve/${revision}/${entry.hfPath}`;
  }

  return {
    FAMILY_ID: familyId,
    HF_REPO_ID: repoIds.length === 1 ? repoIds[0] : null,
    KNOWN_MODELS,
    resolveModel,
    resolveHfBaseUrl,
  };
}
