export const state = {
  // Boot
  phase: 'booting', // booting | ready | loading | generating | error
  bootError: null,

  // Legacy compare/demo state kept explicit for archive-contract surfaces.
  compareHistoryFilter: 'all',
  activeCompareSmokeSampleId: null,
  lastCompareArtifact: null,

  // Model
  modelId: null,
  modelStatus: {}, // { [modelId]: 'available' | 'downloading' | 'stored' | 'loaded' }
  downloadProgress: null, // { percent, downloadedBytes, totalBytes, currentShard, totalShards }

  // Pipeline
  pipeline: null,

  // Generation
  generating: false,
  prefilling: false,
  abortController: null,
  lastRun: null, // { prefillMs, decodeMs, totalTokens, tokPerSec, tokens: [], config }

  // Settings
  preset: 'fast', // 'fast' | 'trace'
  settings: {
    temperature: 0,
    topK: 0,
    topP: 1.0,
    maxTokens: 256,
    readbackInterval: 0,
    runtimeProfile: 'profiles/default',
  },

  // UI toggles
  tokenPressEnabled: false,
  liveTokSec: true,
  xrayEnabled: false,
};
