

// Types and utilities
export { isMoELayerLocal, hasLoggedFusedDownNorm, setLoggedFusedDownNorm } from './types.js';

// Sandwich norm FFN (Gemma 3 style)
export { processFFNWithSandwichNorm } from './sandwich.js';

// Standard FFN (LLaMA style)
export { processFFNStandard } from './standard.js';

// Dense FFN operations
export { runDenseFFNGPU, runDenseFFNWithFusedPostNormGPU } from './dense.js';

// MoE FFN operations
export { runMoEFFNGPU } from './moe.js';
