// Static metadata for the DiffusionGemma family of RDRR artifacts.
// See qwen3.js for the contract.

import { createFamily } from './family.js';

const family = createFamily({
  familyId: 'diffusiongemma',
  hfRepoId: 'Clocksmith/rdrr',
  knownModels: [
    {
      modelId: 'diffusiongemma-26b-a4b-it-q4k-ehf16-af16',
      label: 'DiffusionGemma 26B A4B Instruct (Q4K/F16a)',
      sourceModel: 'google/diffusiongemma-26B-A4B-it',
      hfPath: 'models/diffusiongemma-26b-a4b-it-q4k-ehf16-af16',
      defaultRuntimeProfile: 'profiles/diffusiongemma-26b-a4b-throughput',
      modes: ['diffusion', 'text'],
    },
  ],
});

export const FAMILY_ID = family.FAMILY_ID;
export const HF_REPO_ID = family.HF_REPO_ID;
export const KNOWN_MODELS = family.KNOWN_MODELS;
export const resolveModel = family.resolveModel;
export const resolveHfBaseUrl = family.resolveHfBaseUrl;
