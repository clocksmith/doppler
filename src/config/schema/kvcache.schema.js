// =============================================================================
// Default Config
// =============================================================================

export const DEFAULT_KVCACHE_CONFIG = {
  maxSeqLen: 4096,
  gpuPagedFallbackMaxSeqLen: 4096,
  kvDtype: 'f16',
  forceF32Softcap: false,
  layout: 'contiguous',
  pageSize: 256,
  windowSize: 1024,
  tiering: {
    mode: 'off',
    hotWindow: 1024,
    coldPageSize: 256,
    coldDtype: 'f16',
    compression: {
      mode: 'none',
      blockSize: 1,
    },
    gating: {
      mode: 'auto',
      minAluBwRatio: 0.0,
    },
  },
};

export const PAGED_LAYOUT_SEQ_LEN_THRESHOLD = 8192;
