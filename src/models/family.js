// Shared factory for per-family pointer modules (qwen3, gemma3, gemma4,
// embeddinggemma). Centralizes the resolve helpers so each family file is
// pure data.

export function createFamily({ familyId, hfRepoId, knownModels }) {
  const KNOWN_MODELS = Object.freeze(knownModels.map((m) => Object.freeze({ ...m, modes: Object.freeze([...m.modes]) })));

  function resolveModel(modelId) {
    return KNOWN_MODELS.find((m) => m.modelId === modelId) || null;
  }

  function resolveHfBaseUrl(modelId, revision = 'main') {
    const entry = resolveModel(modelId);
    if (!entry) return null;
    return `https://huggingface.co/${hfRepoId}/resolve/${revision}/${entry.hfPath}`;
  }

  return {
    FAMILY_ID: familyId,
    HF_REPO_ID: hfRepoId,
    KNOWN_MODELS,
    resolveModel,
    resolveHfBaseUrl,
  };
}
