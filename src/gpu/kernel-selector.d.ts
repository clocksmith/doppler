/**
 * Kernel Selector - Backward Compatibility Wrapper
 *
 * This file has been refactored into separate kernel modules in gpu/kernels/.
 * It now serves as a thin re-export wrapper for backward compatibility.
 *
 * For new code, prefer importing from gpu/kernels/index.js directly.
 */

// Re-export everything from the new kernel modules for backward compatibility
export * from './kernels/index.js';
