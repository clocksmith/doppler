/**
 * Kernel Selector - Backward Compatibility Wrapper
 *
 * This file has been refactored into separate kernel modules in gpu/kernels/.
 * It now serves as a thin re-export wrapper for backward compatibility.
 *
 * For new code, prefer importing from gpu/kernels/index.ts directly.
 *
 * Migration completed: kernel-selector.js (3428 lines) split into:
 * - utils.ts - Shared utilities, pipeline cache, KERNEL_CONFIGS
 * - matmul.ts - Matrix multiplication kernels
 * - dequant.ts - Dequantization kernels
 * - attention.ts - Attention kernels
 * - rmsnorm.ts - RMSNorm kernels
 * - softmax.ts - Softmax kernels
 * - rope.ts - RoPE kernels
 * - silu.ts - SiLU activation kernels
 * - gelu.ts - GeLU activation kernels
 * - gather.ts - Gather/embedding lookup kernels
 * - residual.ts - Residual connection kernels
 * - moe.ts - Mixture of Experts kernels
 * - cast.ts - Type casting kernels
 * - index.ts - Barrel export
 */

// Re-export everything from the new kernel modules for backward compatibility
export * from './kernels/index.js';
