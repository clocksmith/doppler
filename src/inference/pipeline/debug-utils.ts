/**
 * Debug utilities for pipeline tracing.
 *
 * Toggleable log categories for surgical debugging without noise.
 * Enable via: setDebugCategories({ embed: true, layer: true })
 *
 * Categories:
 * - embed: Embedding layer output
 * - layer: Per-layer entry/exit, hidden state stats
 * - attn: Attention computation details
 * - ffn: FFN computation details
 * - kv: KV cache operations
 * - logits: Logits computation and top-k
 * - sample: Sampling decisions
 * - io: GPU buffer read/writes
 * - perf: Timing and benchmarks
 * - kernel: Kernel step debugging - inspect tensor state after each kernel (SLOW!)
 * - all: Enable everything (use sparingly)
 *
 * Log format: [CATEGORY] message
 * This enables post-filtering: grep -E "^\[LAYER\]|\[ATTN\]"
 *
 * Kernel Step Debugging:
 * Enable with setDebugCategories({ kernel: true }, { bufferStats: true })
 * or use DEBUG_PRESETS.kernelStep
 *
 * Use dumpTensor() to inspect any GPU buffer's contents.
 * Use dumpKVCache() to inspect KV cache state for a layer.
 *
 * @example
 * // Enable kernel step debugging for layer 0 only
 * setDebugCategories({ kernel: true }, { layers: [0], bufferStats: true });
 *
 * // In pipeline code:
 * if (isKernelDebugEnabled(layerIdx)) {
 *   await dumpTensor(outputBuffer, 'matmul_output', { layerIdx });
 * }
 *
 * @module inference/pipeline/debug-utils
 */

// Re-export facade for backward compatibility
// Implementation split into debug-utils/ submodules

export {
  // Config types and functions
  type DebugCategory,
  type DebugConfig,
  setDebugCategories,
  resetDebugConfig,
  applyPipelineDebugConfig,
  getDebugConfig,
  incrementDecodeStep,
  resetDecodeStep,
  getDecodeStep,
  shouldDebugLayerOutput,
  // Logging functions
  logEmbed,
  logLayer,
  logAttn,
  logFFN,
  logKV,
  logLogits,
  logSample,
  logIO,
  logPerf,
  // Tensor inspection
  type TensorStats,
  dumpTensor,
  dumpTokenVector,
  dumpKVCache,
  logKernelStep,
  isKernelDebugEnabled,
  // Utilities
  f16ToF32,
  decodeReadback,
  getLogitsHealth,
  getBufferStats,
  DEBUG_PRESETS,
} from './debug-utils/index.js';
