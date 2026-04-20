// Static metadata for the Qwen 3.5 family of RDRR artifacts.
// Importing this module does NOT load manifests or shards — it is a
// kilobyte-scale pointer surface for consumers that want a typed,
// tree-shakable way to reference "the Qwen family" without pulling
// the whole catalog or the runtime pipeline.

import { createFamily } from './family.js';

const family = createFamily({
  familyId: 'qwen3',
  hfRepoId: 'Clocksmith/rdrr',
  knownModels: [
    {
      modelId: 'qwen-3-5-0-8b-q4k-ehaf16',
      label: 'Qwen 3.5 0.8B (Q4K)',
      sourceModel: 'Qwen/Qwen3.5-0.8B',
      hfPath: 'models/qwen-3-5-0-8b-q4k-ehaf16',
      defaultRuntimeProfile: 'profiles/qwen-3-5-0-8b-throughput',
      modes: ['text', 'vision'],
    },
    {
      modelId: 'qwen-3-5-2b-q4k-ehaf16',
      label: 'Qwen 3.5 2B (Q4K)',
      sourceModel: 'Qwen/Qwen3.5-2B',
      hfPath: 'models/qwen-3-5-2b-q4k-ehaf16',
      defaultRuntimeProfile: 'profiles/throughput',
      modes: ['text', 'vision'],
    },
  ],
});

export const FAMILY_ID = family.FAMILY_ID;
export const HF_REPO_ID = family.HF_REPO_ID;
export const KNOWN_MODELS = family.KNOWN_MODELS;
export const resolveModel = family.resolveModel;
export const resolveHfBaseUrl = family.resolveHfBaseUrl;
