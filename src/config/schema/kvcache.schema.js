// =============================================================================
// Default Config
// =============================================================================

export const DEFAULT_KVCACHE_CONFIG = {
  maxSeqLen: 4096,
  gpuPagedFallbackMaxSeqLen: 4096,
  kvDtype: 'f16',
  layout: 'contiguous',
  pageSize: 256,
  windowSize: 1024,
};

export const PAGED_LAYOUT_SEQ_LEN_THRESHOLD = 8192;
