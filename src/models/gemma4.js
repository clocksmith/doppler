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
      defaultRuntimeProfile: 'profiles/throughput',
      modes: ['text', 'vision'],
    },
    {
      modelId: 'gemma-4-e2b-it-q4k-ehf16-af32-int4ple',
      label: 'Gemma 4 E2B Instruct (Q4K/F32a/INT4 PLE)',
      sourceModel: 'google/gemma-4-e2b-it',
      hfPath: 'models/gemma-4-e2b-it-q4k-ehf16-af32-int4ple',
      defaultRuntimeProfile: 'profiles/throughput',
      modes: ['text', 'vision'],
    },
    {
      modelId: 'gemma-4-12b-it-text-w4a16-ct-ehf16-af16',
      label: 'Gemma 4 12B Instruct Text (W4A16 QAT/F16a)',
      sourceModel: 'google/gemma-4-12B-it-qat-w4a16-ct',
      hfPath: 'models/gemma-4-12b-it-text-w4a16-ct-ehf16-af16',
      defaultRuntimeProfile: 'profiles/throughput',
      modes: ['text'],
    },
    {
      modelId: 'gemma-4-31b-it-text-q4k-ehf16-af32',
      label: 'Gemma 4 31B Instruct Text (Q4K/F32a)',
      sourceModel: 'google/gemma-4-31B-it',
      hfPath: 'models/gemma-4-31b-it-text-q4k-ehf16-af32',
      defaultRuntimeProfile: 'profiles/throughput',
      modes: ['text'],
    },
    {
      modelId: 'gemma-4-31b-it-text-q4k-ehf16-af16',
      label: 'Gemma 4 31B Instruct Text (Q4K/F16a)',
      sourceModel: 'google/gemma-4-31B-it',
      hfPath: 'models/gemma-4-31b-it-text-q4k-ehf16-af16',
      defaultRuntimeProfile: 'profiles/throughput',
      modes: ['text'],
    },
  ],
});

export const FAMILY_ID = family.FAMILY_ID;
export const HF_REPO_ID = family.HF_REPO_ID;
export const KNOWN_MODELS = family.KNOWN_MODELS;
export const resolveModel = family.resolveModel;
export const resolveHfBaseUrl = family.resolveHfBaseUrl;
