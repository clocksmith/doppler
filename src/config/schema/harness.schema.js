// =============================================================================
// Default Config
// =============================================================================

export const DEFAULT_HARNESS_CONFIG = {
  mode: 'kernels',
  autorun: false,
  skipLoad: false,
  modelId: null,
  trainingBench: {
    ebmRecorded: {
      dims: {
        M: 128,
        K: 512,
        H: 1024,
        O: 1,
      },
    },
  },
};
