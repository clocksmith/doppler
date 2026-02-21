

// Types and utilities
export {
  shouldDebugLayer,
  markStageLogged,
  releaseOrTrack,
  getQKNormOnesBuffer,
} from './types.js';

// Run (immediate submission)
export { runLayerAttentionGPU } from './run.js';

// Record (batched submission)
export { recordLayerAttentionGPU } from './record.js';
export { runAttentionBDPA } from '../../../../gpu/kernel-selector.js';
export { recordAttentionBDPA } from '../../../../gpu/kernel-selector.js';
