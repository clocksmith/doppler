// Re-export everything from the new kernel modules for backward compatibility
export * from './kernels/index.js';
export {
    runAttentionBDPA,
    runAttentionTiered,
    recordAttentionTiered,
    runAttentionTieredQuant,
    recordAttentionTieredQuant,
    recordAttentionBDPA,
    runAttentionContiguousQuant,
    recordAttentionContiguousQuant,
} from './kernels/attention.js';
