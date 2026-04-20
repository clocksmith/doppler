// Static metadata for the EmbeddingGemma family of RDRR artifacts.

import { createFamily } from './family.js';

const family = createFamily({
  familyId: 'embeddinggemma',
  hfRepoId: 'Clocksmith/rdrr',
  knownModels: [
    {
      modelId: 'google-embeddinggemma-300m-q4k-ehf16-af32',
      label: 'EmbeddingGemma 300M (Q4K)',
      sourceModel: 'google/embeddinggemma-300m',
      hfPath: 'models/google-embeddinggemma-300m-q4k-ehf16-af32',
      defaultRuntimeProfile: 'profiles/throughput',
      modes: ['embedding'],
    },
  ],
});

export const FAMILY_ID = family.FAMILY_ID;
export const HF_REPO_ID = family.HF_REPO_ID;
export const KNOWN_MODELS = family.KNOWN_MODELS;
export const resolveModel = family.resolveModel;
export const resolveHfBaseUrl = family.resolveHfBaseUrl;
