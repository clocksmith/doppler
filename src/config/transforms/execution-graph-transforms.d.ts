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
  };
  platform: {
    id: string;
    vendor: string;
    architecture: string;
  };
  activationDtype: 'f16' | 'f32';
  kvDtype: 'f16' | 'f32';
  modelId?: string;
}

/**
 * An execution graph kernel entry from manifest.inference.execution.kernels.
 */
export interface ExecutionKernelEntry {
  kernel: string;
  entry: string;
  digest: string | null;
  constants?: Record<string, unknown>;
}

/**
 * The execution-v1 graph structure from manifest.inference.execution.
 */
export interface ExecutionGraph {
  kernels: Record<string, ExecutionKernelEntry>;
  preLayer: unknown[][];
  decode: unknown[][];
  prefill: unknown[][];
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
 * Compose multiple transforms into a single transform.
 * Applies left-to-right, skipping transforms that return null.
 */
export declare function composeTransforms(...transforms: ExecutionGraphTransform[]): ExecutionGraphTransform;

/**
 * Registry mapping transform names to functions.
 */
export declare const TRANSFORMS: Record<string, ExecutionGraphTransform>;
