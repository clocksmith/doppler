// Static metadata for the Qwen 3.5 family of RDRR artifacts.
// Importing this module does NOT load manifests or shards — it is a
// kilobyte-scale pointer surface for consumers that want a typed,
// tree-shakable way to reference "the Qwen family" without pulling
// the whole catalog or the runtime pipeline.

export const FAMILY_ID = 'qwen3';
export const HF_REPO_ID = 'Clocksmith/rdrr';

export const KNOWN_MODELS = Object.freeze([
  Object.freeze({
    modelId: 'qwen-3-5-0-8b-q4k-ehaf16',
    label: 'Qwen 3.5 0.8B (Q4K)',
    sourceModel: 'Qwen/Qwen3.5-0.8B',
    hfPath: 'models/qwen-3-5-0-8b-q4k-ehaf16',
    defaultRuntimeProfile: 'profiles/qwen-3-5-0-8b-throughput',
    modes: Object.freeze(['text', 'vision']),
  }),
  Object.freeze({
    modelId: 'qwen-3-5-2b-q4k-ehaf16',
    label: 'Qwen 3.5 2B (Q4K)',
    sourceModel: 'Qwen/Qwen3.5-2B',
    hfPath: 'models/qwen-3-5-2b-q4k-ehaf16',
    defaultRuntimeProfile: 'profiles/throughput',
    modes: Object.freeze(['text', 'vision']),
  }),
]);

export function resolveModel(modelId) {
  return KNOWN_MODELS.find((m) => m.modelId === modelId) || null;
}

export function resolveHfBaseUrl(modelId, revision = 'main') {
  const entry = resolveModel(modelId);
  if (!entry) return null;
  return `https://huggingface.co/${HF_REPO_ID}/resolve/${revision}/${entry.hfPath}`;
}
