/**
 * Kernel Registry Schema Definitions
 *
 * Defines the structure for kernel metadata: variants, bindings, uniforms,
 * workgroup sizes, and GPU feature requirements.
 *
 * @module config/schema/kernel-registry
 */

// =============================================================================
// GPU Feature Requirements
// =============================================================================

/**
 * GPU features that a kernel variant may require.
 * These map to WebGPU adapter features.
 */
export type GpuFeature = 'shader-f16' | 'subgroups' | 'subgroups-f16';

// =============================================================================
// Binding Types
// =============================================================================

/**
 * WebGPU buffer binding types.
 */
export type BindingType = 'uniform' | 'storage' | 'read-only-storage';

/**
 * A single buffer binding in a kernel's bind group layout.
 */
export interface BindingSchema {
  /** Binding index in the bind group */
  index: number;

  /** Human-readable name for debugging */
  name: string;

  /** Buffer type (uniform, storage, read-only-storage) */
  type: BindingType;

  /** Optional: whether this binding is optional (can be null) */
  optional?: boolean;
}

// =============================================================================
// Uniform Fields
// =============================================================================

/**
 * Primitive types for uniform struct fields.
 */
export type UniformFieldType = 'u32' | 'i32' | 'f32' | 'f16';

/**
 * A field in a uniform buffer struct.
 */
export interface UniformFieldSchema {
  /** Field name (matches WGSL struct field) */
  name: string;

  /** Data type */
  type: UniformFieldType;

  /** Byte offset in the uniform buffer */
  offset: number;
}

/**
 * Complete uniform buffer schema for a kernel.
 */
export interface UniformsSchema {
  /** Total size in bytes (must be 16-byte aligned for WebGPU) */
  size: number;

  /** Fields in the uniform struct */
  fields: UniformFieldSchema[];
}

// =============================================================================
// Kernel Variant
// =============================================================================

/**
 * WGSL override constants that can be set at pipeline creation.
 */
export interface WgslOverridesSchema {
  [constantName: string]: number;
}

/**
 * A single kernel variant (e.g., matmul/f16, matmul/q4_fused).
 */
export interface KernelVariantSchema {
  /** WGSL shader filename (relative to kernels directory) */
  wgsl: string;

  /** Entry point function name in the shader */
  entryPoint: string;

  /** Workgroup size [x, y, z] */
  workgroup: [number, number, number];

  /** GPU features required to use this variant */
  requires?: GpuFeature[];

  /** Estimated shared memory usage in bytes */
  sharedMemory?: number;

  /** WGSL override constants to set at pipeline creation */
  wgslOverrides?: WgslOverridesSchema;

  /** Additional bindings beyond the base operation bindings */
  bindingsOverride?: BindingSchema[];

  /** Override uniform schema (if different from base) */
  uniformsOverride?: UniformsSchema;

  /** Human-readable description */
  description?: string;
}

// =============================================================================
// Operation
// =============================================================================

/**
 * An operation (e.g., matmul, attention, rmsnorm) with all its variants.
 */
export interface OperationSchema {
  /** Base bindings shared by all variants */
  baseBindings: BindingSchema[];

  /** Base uniforms shared by all variants */
  baseUniforms: UniformsSchema;

  /** Available variants for this operation */
  variants: Record<string, KernelVariantSchema>;

  /** Human-readable description of the operation */
  description?: string;
}

// =============================================================================
// Kernel Registry
// =============================================================================

/**
 * Complete kernel registry with all operations and their variants.
 */
export interface KernelRegistrySchema {
  /** Schema version for compatibility checking */
  version: string;

  /** All kernel operations */
  operations: Record<string, OperationSchema>;
}

// =============================================================================
// Resolved Types (after merging base + variant)
// =============================================================================

/**
 * Fully resolved kernel configuration after merging base and variant.
 */
export interface ResolvedKernelConfig {
  /** Operation name */
  operation: string;

  /** Variant name */
  variant: string;

  /** WGSL shader filename */
  wgsl: string;

  /** Entry point function name */
  entryPoint: string;

  /** Workgroup size [x, y, z] */
  workgroup: [number, number, number];

  /** GPU features required */
  requires: GpuFeature[];

  /** All bindings (base + override merged) */
  bindings: BindingSchema[];

  /** Uniform schema (base or overridden) */
  uniforms: UniformsSchema;

  /** WGSL override constants */
  wgslOverrides: WgslOverridesSchema;

  /** Estimated shared memory usage in bytes */
  sharedMemory: number;
}

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Merge base and variant bindings.
 * Variant bindings with matching indices override base bindings.
 */
export function mergeBindings(
  base: BindingSchema[],
  override?: BindingSchema[]
): BindingSchema[] {
  if (!override || override.length === 0) {
    return [...base];
  }

  const result = [...base];
  for (const binding of override) {
    const existingIdx = result.findIndex(b => b.index === binding.index);
    if (existingIdx >= 0) {
      result[existingIdx] = binding;
    } else {
      result.push(binding);
    }
  }

  return result.sort((a, b) => a.index - b.index);
}

/**
 * Resolve a kernel variant to a complete configuration.
 */
export function resolveKernelConfig(
  operation: string,
  variant: string,
  opSchema: OperationSchema,
  variantSchema: KernelVariantSchema
): ResolvedKernelConfig {
  return {
    operation,
    variant,
    wgsl: variantSchema.wgsl,
    entryPoint: variantSchema.entryPoint,
    workgroup: variantSchema.workgroup,
    requires: variantSchema.requires ?? [],
    bindings: mergeBindings(opSchema.baseBindings, variantSchema.bindingsOverride),
    uniforms: variantSchema.uniformsOverride ?? opSchema.baseUniforms,
    wgslOverrides: variantSchema.wgslOverrides ?? {},
    sharedMemory: variantSchema.sharedMemory ?? 0,
  };
}
