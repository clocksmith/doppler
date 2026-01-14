/**
 * Kernel Path Schema
 *
 * Defines explicit, ordered kernel dispatch sequences for inference.
 * Replaces the implicit q4kStrategy/fusedFFNQ4K configuration.
 *
 * A kernel path is a complete specification of:
 * - Which kernels run
 * - In what order
 * - With what override constants
 * - With what entry points
 *
 * @module config/schema/kernel-path
 */

/**
 * A single kernel dispatch in the path.
 */
export interface KernelStepSchema {
  /**
   * Logical operation name (for debugging/tracing).
   * Examples: 'rmsnorm', 'q_proj', 'attention', 'ffn_fused'
   */
  op: string;

  /**
   * Kernel file name (without path).
   * Examples: 'rmsnorm.wgsl', 'fused_matmul_q4.wgsl'
   */
  kernel: string;

  /**
   * Entry point function name.
   * @default 'main'
   */
  entry?: string;

  /**
   * Override constants for pipeline creation.
   * These are compile-time constants that affect code generation.
   */
  constants?: Record<string, number | boolean>;

  /**
   * Weight buffer reference (for matmul ops).
   * Uses template syntax: 'layer.{L}.self_attn.q_proj'
   * {L} is replaced with layer index at runtime.
   */
  weights?: string;

  /**
   * Input buffer slot name.
   * @default 'hidden_state'
   */
  input?: string;

  /**
   * Output buffer slot name.
   * @default 'hidden_state'
   */
  output?: string;
}

/**
 * Kernel sequence for a single transformer layer.
 */
export interface LayerKernelPathSchema {
  /** Ordered list of kernel dispatches */
  steps: KernelStepSchema[];
}

/**
 * Override for specific layers (e.g., first/last layer differences).
 */
export interface LayerOverrideSchema {
  /** Layer indices this override applies to */
  layers: number[];

  /** Steps to use instead of default */
  steps: KernelStepSchema[];
}

/**
 * Complete kernel path specification for a model.
 */
export interface KernelPathSchema {
  /** Path identifier */
  id: string;

  /** Human-readable name */
  name: string;

  /** Description of this path's characteristics */
  description?: string;

  /** Activation dtype for this path (e.g., 'f16', 'f32') */
  activationDtype: string;

  /**
   * Prefill phase kernel sequence (M > 1).
   * If not specified, uses decode with batched variants.
   */
  prefill?: LayerKernelPathSchema;

  /**
   * Decode phase kernel sequence (M = 1).
   */
  decode: LayerKernelPathSchema;

  /**
   * Layer-specific overrides.
   * For models with different first/last layer behavior.
   */
  layerOverrides?: LayerOverrideSchema[];

  /**
   * Pre-layer operations (embedding lookup, initial norm).
   */
  preLayer?: KernelStepSchema[];

  /**
   * Post-layer operations (final norm, LM head).
   */
  postLayer?: KernelStepSchema[];

  /**
   * Sampling kernels.
   */
  sampling?: KernelStepSchema[];
}

/**
 * Built-in kernel path identifiers.
 * These are the known preset IDs, but custom presets can also be registered.
 */
export type BuiltinKernelPathId =
  | 'gemma2-q4k-fused-f16a'   // Gemma 2 Q4K weights, fused matmul, F16 activations
  | 'gemma2-q4k-fused-f32a'   // Gemma 2 Q4K weights, fused matmul, F32 activations
  | 'gemma2-q4k-dequant-f16a' // Gemma 2 Q4K -> F16 dequant, F16 activations
  | 'gemma2-q4k-dequant-f32a' // Gemma 2 Q4K -> F32 dequant, F32 activations
  | 'gemma2-f16-f16a'         // Gemma 2 F16 weights, F16 activations
  | 'gemma2-f16-f32a';        // Gemma 2 F16 weights, F32 activations

/**
 * Kernel path reference - preset ID (string) or inline path schema.
 * Accepts any string for custom preset IDs, not just built-in IDs.
 */
export type KernelPathRef = string | KernelPathSchema;

/** Default entry point */
export declare const DEFAULT_ENTRY: string;

/** Default input slot */
export declare const DEFAULT_INPUT: string;

/** Default output slot */
export declare const DEFAULT_OUTPUT: string;
