/**
 * Kernel Plan Schema
 *
 * Defines config-driven kernel pipeline ordering and variant overrides.
 *
 * @module config/schema/kernel-plan
 */

import type { LayerPipelineSchema } from './inference.schema.js';

// =============================================================================
// Kernel Variant Overrides
// =============================================================================

/**
 * Variant override for a kernel operation.
 *
 * - default: apply to all uses of the operation
 * - prefill/decode: phase-specific override (attention)
 * - roles: per-role override (matmul, rmsnorm, etc.)
 */
export interface KernelVariantOverrideSchema {
  /** Default variant for the operation */
  default?: string;
  /** Prefill-specific variant override */
  prefill?: string;
  /** Decode-specific variant override */
  decode?: string;
  /** Role-specific variant overrides */
  roles?: Record<string, string>;
}

/**
 * Variant overrides for all kernel operations.
 */
export interface KernelVariantOverridesSchema {
  attention?: KernelVariantOverrideSchema;
  matmul?: KernelVariantOverrideSchema;
  rmsnorm?: KernelVariantOverrideSchema;
  rope?: KernelVariantOverrideSchema;
  softmax?: KernelVariantOverrideSchema;
  silu?: KernelVariantOverrideSchema;
  gelu?: KernelVariantOverrideSchema;
  gather?: KernelVariantOverrideSchema;
  residual?: KernelVariantOverrideSchema;
  cast?: KernelVariantOverrideSchema;
  sample?: KernelVariantOverrideSchema;
  dequant?: KernelVariantOverrideSchema;
  splitQKV?: KernelVariantOverrideSchema;
  fusedFFN?: KernelVariantOverrideSchema;
  fusedMatmulResidual?: KernelVariantOverrideSchema;
  fusedMatmulRmsnorm?: KernelVariantOverrideSchema;
  moe?: KernelVariantOverrideSchema;
}

// =============================================================================
// Kernel Plan
// =============================================================================

/** Q4K strategy for weight handling */
export type Q4KStrategy = 'auto' | 'fused_q4k' | 'dequant_f16' | 'dequant_f32';

/**
 * Kernel plan for configuring pipeline order and variant selection.
 */
export interface KernelPlanSchema {
  /** Merge mode when combining model + runtime plans */
  mode?: 'replace' | 'patch';
  /** Layer pipeline plan (order of ops inside each layer) */
  layerPipeline?: LayerPipelineSchema | null;
  /** Variant overrides for kernel operations */
  variants?: KernelVariantOverridesSchema;
  /** Q4K strategy for weight loading and matmul selection */
  q4kStrategy?: Q4KStrategy;
  /** Throw on invalid/unavailable variants instead of falling back */
  strict?: boolean;
}
