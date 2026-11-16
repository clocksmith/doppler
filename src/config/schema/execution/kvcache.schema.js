// =============================================================================
// Default Config
// =============================================================================

// maxSeqLen has no code default: the manifest (inference.session.kvcache, stamped
// from the conversion config), the per-model declaration in
// config/runtime/model-sessions.json, or an explicit runtime profile/override
// declares it, and KV allocation fails fast when none does.
export const DEFAULT_KVCACHE_CONFIG = {
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
      bitWidth: 4,
      prodMode: false,
    },
    gating: {
      mode: 'auto',
      minAluBwRatio: 0.0,
    },
  },
  // Contiguous quantized KV cache controls (for full-attention models)
  // Activates when layout is auto-resolved to 'contiguous_quantized' or set explicitly.
  quantization: {
    mode: 'none',       // none | turboquant | turboquant_prod
    bitWidth: 4,
    prodMode: false,
  },
};

export const PAGED_LAYOUT_SEQ_LEN_THRESHOLD = 8192;

// Returns the declared session KV limit or fails with the places that may declare it.
export function requireDeclaredKVMaxSeqLen(value) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(
      `KV cache maxSeqLen is not declared (got ${String(value)}). Declare ` +
      'inference.session.kvcache.maxSeqLen in the conversion config (stamped into the manifest), ' +
      'add the published model to src/config/runtime/model-sessions.json, ' +
      'or set runtime.inference.session.kvcache.maxSeqLen in a runtime profile or override.'
    );
  }
  return value;
}
