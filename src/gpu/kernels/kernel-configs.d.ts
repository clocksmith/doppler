/**
 * Kernel Configurations - Table-driven kernel metadata
 *
 * Contains all kernel configurations including shader files, entry points,
 * workgroup sizes, and feature requirements.
 *
 * @module gpu/kernels/kernel-configs
 */

// ============================================================================
// Types
// ============================================================================

/** Variant-specific metadata for table-driven kernel configuration */
export interface VariantMetadata {
  /** Columns processed per workgroup (matmul multicol variants) */
  colsPerWg?: number;
  /** Tile size for M dimension (batched matmul variants) */
  tileM?: number;
  /** Output buffer binding index (gather F16 output variants) */
  outputBinding?: number;
  /** Maximum KV length for chunked attention */
  maxKVLen?: number;
}

/** Kernel configuration */
export interface KernelConfig {
  shaderFile: string;
  entryPoint: string;
  workgroupSize: [number, number, number];
  requires: string[];
  validate?: (seqLen: number, numHeads: number, headDim: number) => void;
  /** Output dtype for variants that output to specific precision */
  outputDtype?: 'f16' | 'f32';
  /** Tile size for attention/matmul kernels */
  tileSize?: number;
  /** Additional variant-specific configuration */
  variantMetadata?: VariantMetadata;
}

// ============================================================================
// Kernel Configurations
// ============================================================================

/** All kernel configurations by operation and variant */
export declare const KERNEL_CONFIGS: Record<string, Record<string, KernelConfig>>;

// ============================================================================
// Config Helpers
// ============================================================================

/**
 * Get kernel configuration
 */
export declare function getKernelConfig(operation: string, variant: string): KernelConfig;

/**
 * Set a validator function on a kernel config.
 * Used to set attention validators after import to avoid circular dependencies.
 */
export declare function setKernelValidator(
  operation: string,
  variant: string,
  validator: (seqLen: number, numHeads: number, headDim: number) => void
): void;
