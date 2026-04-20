// Static metadata for the Gemma 3 family of RDRR artifacts.
// See qwen3.js for the contract.

import { createFamily } from './family.js';

const family = createFamily({
  familyId: 'gemma3',
  hfRepoId: 'Clocksmith/rdrr',
  knownModels: [
    {
      modelId: 'gemma-3-270m-it-q4k-ehf16-af32',
      label: 'Gemma 3 270M Instruct (Q4K/F32a)',
      sourceModel: 'google/gemma-3-270m-it',
      hfPath: 'models/gemma-3-270m-it-q4k-ehf16-af32',
      defaultRuntimeProfile: 'profiles/throughput',
      modes: ['text', 'vision'],
    },
    {
      modelId: 'gemma-3-1b-it-q4k-ehf16-af32',
      label: 'Gemma 3 1B Instruct (Q4K/F32a)',
      sourceModel: 'google/gemma-3-1b-it',
      hfPath: 'models/gemma-3-1b-it-q4k-ehf16-af32',
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
