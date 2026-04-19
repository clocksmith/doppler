export function normalizeModelId(modelId) {
  return String(modelId).replace(/[^a-zA-Z0-9_-]/g, '_');
}
