/**
 * Execution Graph Transforms
 *
 * Pure functions that transform execution-v1 graphs based on GPU
 * capabilities and platform constraints. Each transform either returns
 * a modified graph or null when not applicable.
 *
 * @module config/transforms/execution-graph-transforms
 */

/**
 * Context passed to all execution graph transforms.
 */
export interface TransformContext {
  capabilities: {
    hasSubgroups: boolean;
    hasF16: boolean;
    hasSubgroupsF16: boolean;
    maxWorkgroupStorageSize?: number;
  };
  platform: {
    id: string;
    vendor: string;
    architecture: string;
  };
  activationDtype: 'f16' | 'f32';
  kvDtype: 'f16' | 'f32';
  modelId?: string;
  layerTypes?: string[] | null;
}

export interface ExecutionKernelPrecision {
  activationDtype?: 'f16' | 'f32';
  kvDtype?: 'f16' | 'f32';
  inputDtype?: 'f16' | 'f32';
  outputDtype?: 'f16' | 'f32';
}

/**
 * An execution graph kernel entry from manifest.inference.execution.kernels.
 */
export interface ExecutionKernelEntry {
  kernel: string;
  entry: string;
  digest: string | null;
  constants?: Record<string, unknown>;
  precision?: ExecutionKernelPrecision;
}

export interface ExecutionGraphLayerGroup {
  layers: number[];
  steps: unknown[][];
}

export type ExecutionGraphLayerEntry = unknown[] | ExecutionGraphLayerGroup;

/**
 * The execution-v1 graph structure from manifest.inference.execution.
 */
export interface ExecutionGraph {
  kernels: Record<string, ExecutionKernelEntry>;
  preLayer: unknown[][];
  decode: ExecutionGraphLayerEntry[];
  prefill: ExecutionGraphLayerEntry[];
  postLayer: unknown[][];
  policies?: Record<string, unknown>;
}

/**
 * A pure function that transforms an execution graph.
 * Returns the modified graph, or null if the transform is not applicable.
 */
export type ExecutionGraphTransform = (graph: ExecutionGraph, ctx: TransformContext) => ExecutionGraph | null;

/**
 * Remove subgroup-dependent kernels from the execution graph,
 * replacing with scalar/tiled equivalents.
 */
export declare function removeSubgroups(graph: ExecutionGraph, ctx: TransformContext): ExecutionGraph | null;

/**
 * Widen all f16-activation kernels to f32 equivalents.
 * Returns null if the graph uses fused f16 FFN (not transformable).
 */
export declare function widenToF32Activations(graph: ExecutionGraph, ctx: TransformContext): ExecutionGraph | null;

/**
 * Swap prefill attention kernel between streaming and small-tile variants.
 */
export declare function swapPrefillAttention(
  graph: ExecutionGraph,
  ctx: TransformContext,
  options: { from: string; to: string }
): ExecutionGraph | null;

/**
 * Replace eligible prefill f16kv attention kernels with the fixed 256-dim
 * shared-block variant when the graph is eligible.
 */
export declare function useHead256PrefillAttention(
  graph: ExecutionGraph,
  ctx: TransformContext
): ExecutionGraph | null;

/**
 * Replace projection matmul kernels with f32-weight variants for numeric debugging.
 */
export declare function widenProjectionWeightsToF32(graph: ExecutionGraph, ctx: TransformContext): ExecutionGraph | null;

/**
 * Replace dense Q4K prefill projections with explicit Q4-native prefill kernels
 * when the graph already exposes a compatible fused Q4 decode kernel.
 */
export declare function remapDenseQ4KPrefillToQ4Native(
  graph: ExecutionGraph,
  ctx: TransformContext
): ExecutionGraph | null;

/**
 * Replace fused Q4K prefill projections with dense tiled matmul kernels while
 * leaving decode unchanged.
 */
export declare function remapQ4KPrefillToDense(
  graph: ExecutionGraph,
  ctx: TransformContext
): ExecutionGraph | null;

/**
 * Mark Qwen linear-attention decode q/o projections as f16 for targeted
 * Apple/WebGPU decode throughput work while keeping full-attention layers on
 * the manifest-owned f32 activation contract.
 */
export declare function useLinearDecodeProjectionF16(
  graph: ExecutionGraph,
  ctx: TransformContext
): ExecutionGraph | null;

/**
 * Replace fused Q4K decode projection kernels with GEMV subgroup variants.
 * On pre-dequantized f16 weights the GEMV path is ~2.3x faster than the fused
 * Q4K kernel for M=1 decode.
 */
export declare function remapQ4KDecodeToGemv(
  graph: ExecutionGraph,
  ctx: TransformContext
): ExecutionGraph | null;

/**
 * Replace ONLY attention-projection decode kernels (q/k/v/o_proj) with GEMV
 * subgroup variants, leaving FFN projections as fused Q4K.
 *
 * Diagnostic transform for isolating GEMV correctness regressions between the
 * attention and FFN decode paths.
 */
export declare function remapQ4KDecodeAttentionToGemv(
  graph: ExecutionGraph,
  ctx: TransformContext
): ExecutionGraph | null;

/**
 * Replace ONLY FFN-projection decode kernels (gate/up/down_proj) with GEMV
 * subgroup variants, leaving attention projections as fused Q4K.
 *
 * Diagnostic complement to `remapQ4KDecodeAttentionToGemv` for isolating
 * GEMV correctness regressions to the FFN decode path.
 */
export declare function remapQ4KDecodeFFNToGemv(
  graph: ExecutionGraph,
  ctx: TransformContext
): ExecutionGraph | null;

/**
 * Narrow selected Qwen decode FFN + lm_head matmuls onto explicit f16 kernels
 * while keeping the manifest-owned activation contract intact.
 */
export declare function useQwenDecodeF16Matmuls(
  graph: ExecutionGraph,
  ctx: TransformContext
): ExecutionGraph | null;

/**
 * Compose multiple transforms into a single transform.
 * Applies left-to-right, skipping transforms that return null.
 */
export declare function composeTransforms(...transforms: ExecutionGraphTransform[]): ExecutionGraphTransform;

/**
 * Registry mapping transform names to functions.
 */
export declare const TRANSFORMS: Record<string, ExecutionGraphTransform>;
