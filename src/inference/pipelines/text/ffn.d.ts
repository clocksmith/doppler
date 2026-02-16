/**
 * Feed-Forward Network (FFN) Operations
 *
 * This module handles the FFN block of the Transformer layer, including:
 * - Standard Dense FFN (Gate/Up -> Act -> Down)
 * - MoE FFN (Routing -> Expert -> Down)
 * - Sandwich Norm variations
 * - Fused optimizations (Fused FFN, Fused Down+Norm)
 *
 * Re-export facade for backward compatibility.
 * Implementation split into submodules under ./ffn/
 *
 * @module inference/pipelines/text/ffn
 */

// Re-export all public API from submodules
export {
  // Types and utilities
  isMoELayerLocal,
  hasLoggedFusedDownNorm,
  setLoggedFusedDownNorm,
  // Sandwich norm FFN (Gemma 3 style)
  processFFNWithSandwichNorm,
  // Standard FFN (LLaMA style)
  processFFNStandard,
  // Dense FFN operations
  runDenseFFNGPU,
  runDenseFFNWithFusedPostNormGPU,
  // MoE FFN operations
  runMoEFFNGPU,
} from './ffn/index.js';
