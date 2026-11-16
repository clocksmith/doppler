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
   * Optional per-step precision contract. Used by execution-v1-derived kernel
   * paths to narrow or widen only specific ops without changing the global
   * activation/KV contract.
   */
  precision?: {
    activationDtype?: 'f16' | 'f32';
    kvDtype?: 'f16' | 'f32';
    inputDtype?: 'f16' | 'f32';
    outputDtype?: 'f16' | 'f32';
  };

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

  /**
   * Legacy all-phase override steps. When present, these replace both decode
   * and prefill defaults.
   */
  steps?: KernelStepSchema[];

  /**
   * Phase-specific decode override. When omitted, decode falls back to
   * `steps` or the base decode path.
   */
  decode?: LayerKernelPathSchema;

  /**
   * Phase-specific prefill override. When omitted, prefill falls back to
   * `steps` or the base prefill path.
   */
  prefill?: LayerKernelPathSchema;
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
   * Output dtype for kernels in this path (e.g., 'f16', 'f32').
   * Defaults to `activationDtype` when omitted.
   */
  outputDtype?: string;

  /** KV cache dtype for this path; defaults to activationDtype when omitted. */
  kvDtype?: string;

  /**
   * Explicit alternate-plan target used only when finiteness recovery is opted in.
   * Required for inline/generated kernel paths that do not have a stable registry id.
   */
  finitenessFallbackKernelPathId?: string;

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
 * Kernel path reference.
 *
 * Registry string IDs were removed with the execution-v1 migration. Runtime
 * and manifest surfaces now accept only inline kernel path objects generated
 * from a pinned execution graph, or null for no explicit override.
 */
export type KernelPathRef = KernelPathSchema | null;

/** Default entry point */
export declare const DEFAULT_ENTRY: string;

/** Default input slot */
export declare const DEFAULT_INPUT: string;

/** Default output slot */
export declare const DEFAULT_OUTPUT: string;
