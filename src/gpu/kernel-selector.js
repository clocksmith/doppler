// Re-export everything from the new kernel modules for backward compatibility
export * from './kernels/index.js';
export {
    runAttentionBDPA,
    runAttentionTiered,
    recordAttentionTiered,
    runAttentionTieredQuant,
    recordAttentionTieredQuant,
    recordAttentionBDPA
} from './kernels/attention.js';
