// Static metadata for the Gemma 4 family of RDRR artifacts.
// See qwen3.js for the contract.

import { createFamily } from './family.js';

const family = createFamily({
  familyId: 'gemma4',
  hfRepoId: 'Clocksmith/rdrr',
  knownModels: [
    {
      modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
      label: 'Gemma 4 E2B Instruct (Q4K/F32a)',
      sourceModel: 'google/gemma-4-e2b-it',
      hfPath: 'models/gemma-4-e2b-it-q4k-ehf16-af32',
      defaultRuntimeProfile: 'profiles/gemma4-e2b-throughput',
      modes: ['text', 'vision'],
    },
    {
      modelId: 'gemma-4-e2b-it-q4k-ehf16-af32-int4ple',
      label: 'Gemma 4 E2B Instruct (Q4K/F32a/INT4 PLE)',
      sourceModel: 'google/gemma-4-e2b-it',
      hfPath: 'models/gemma-4-e2b-it-q4k-ehf16-af32-int4ple',
      defaultRuntimeProfile: 'profiles/gemma4-e2b-throughput',
      modes: ['text', 'vision'],
    },
  ],
});

export const FAMILY_ID = family.FAMILY_ID;
export const HF_REPO_ID = family.HF_REPO_ID;
export const KNOWN_MODELS = family.KNOWN_MODELS;
export const resolveModel = family.resolveModel;
export const resolveHfBaseUrl = family.resolveHfBaseUrl;
