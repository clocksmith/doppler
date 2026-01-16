

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
