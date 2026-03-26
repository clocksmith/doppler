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
  // Basis-decomposed paged KV cache controls (BDPA experimental layout)
  bdpaVocabSize: 2048,
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
  // Contiguous quantized KV cache controls (for full-attention models)
  // Activates when layout is auto-resolved to 'contiguous_quantized' or set explicitly.
  quantization: {
    mode: 'none',       // none | turboquant | turboquant_prod | turboquant_outlier
    bitWidth: 4,
    prodMode: false,
  },
};

export const PAGED_LAYOUT_SEQ_LEN_THRESHOLD = 8192;

const VALID_KV_DTYPES = new Set(['f16', 'f32']);

export function validateKvCacheDtype(dtype) {
  if (dtype != null && !VALID_KV_DTYPES.has(dtype)) {
    throw new Error(
      `DopplerConfigError: kvcache.kvDtype must be "f16" or "f32"; got ${JSON.stringify(dtype)}.`
    );
  }
}
