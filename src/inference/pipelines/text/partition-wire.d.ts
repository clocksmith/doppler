/** Borrowed f32 GPU activation -> owned finite f16 wire bytes through declared GPU casts.
 * Throws on nonfinite/out-of-range input, cancellation or failed completion. */
export function readPartitionF16Boundary(input: GPUBuffer, shape: readonly number[], signal: AbortSignal): Promise<ArrayBuffer>;
