/**
 * Dispatch Helpers - Simplified GPU kernel dispatch
 *
 * Provides helpers to reduce boilerplate for common dispatch patterns:
 * - Single submit dispatch
 * - CommandRecorder dispatch (batched)
 * - Multi-dimensional dispatch
 */

import type { CommandRecorder } from '../command-recorder.js';

/**
 * Dispatch a single compute pass and submit immediately
 * Use for standalone kernels that don't participate in batching
 */
export declare function dispatch(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  workgroups: number | [number, number, number],
  label?: string
): void;

/**
 * Record a compute pass to a CommandRecorder (no submit)
 * Use for kernels in the batched pipeline path
 */
export declare function recordDispatch(
  recorder: CommandRecorder,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  workgroups: number | [number, number, number],
  label?: string
): void;

/**
 * Dispatch a single compute pass using an indirect dispatch buffer
 * Use when workgroup counts are produced on GPU
 */
export declare function dispatchIndirect(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  indirectBuffer: GPUBuffer,
  indirectOffset?: number,
  label?: string
): void;

/**
 * Record an indirect dispatch into a CommandRecorder (no submit)
 */
export declare function recordDispatchIndirect(
  recorder: CommandRecorder,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  indirectBuffer: GPUBuffer,
  indirectOffset?: number,
  label?: string
): void;

/**
 * Dispatch with multiple bind groups
 * For kernels that use multiple bind group sets
 */
export declare function dispatchMultiBindGroup(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindGroups: GPUBindGroup[],
  workgroups: number | [number, number, number],
  label?: string
): void;

/**
 * Calculate workgroup count for 1D dispatch
 * @param totalThreads - Total number of threads needed
 * @param workgroupSize - Threads per workgroup (default: 256)
 * @returns Number of workgroups (rounded up)
 */
export declare function calculateWorkgroups1D(
  totalThreads: number,
  workgroupSize?: number
): number;

/**
 * Calculate workgroup count for 2D dispatch
 * @param width - Width dimension (e.g., matrix columns)
 * @param height - Height dimension (e.g., matrix rows)
 * @param tileSize - Tile size per workgroup (default: 16)
 * @returns [workgroupsX, workgroupsY]
 */
export declare function calculateWorkgroups2D(
  width: number,
  height: number,
  tileSize?: number
): [number, number];

/**
 * Calculate workgroup count for 3D dispatch
 * @param width - Width dimension
 * @param height - Height dimension
 * @param depth - Depth dimension
 * @param tileSizeX - Tile size in X (default: 16)
 * @param tileSizeY - Tile size in Y (default: 16)
 * @param tileSizeZ - Tile size in Z (default: 1)
 * @returns [workgroupsX, workgroupsY, workgroupsZ]
 */
export declare function calculateWorkgroups3D(
  width: number,
  height: number,
  depth: number,
  tileSizeX?: number,
  tileSizeY?: number,
  tileSizeZ?: number
): [number, number, number];

/**
 * Dispatch options for advanced use cases
 */
export interface DispatchOptions {
  /** Custom label for encoder and pass */
  label?: string;

  /** Bind groups (default: single group at index 0) */
  bindGroups?: GPUBindGroup[];

  /** Push constants (if supported) */
  pushConstants?: ArrayBuffer;

  /** Timestamp queries (if available) */
  timestampWrites?: GPUComputePassTimestampWrites;
}

/**
 * Advanced dispatch with full control
 * Supports push constants, timestamps, and multiple bind groups
 */
export declare function dispatchAdvanced(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  workgroups: number | [number, number, number],
  options?: DispatchOptions
): void;

/**
 * Batch multiple dispatches in a single command buffer
 * Useful for multi-kernel operations that should be submitted together
 */
export declare function dispatchBatch(
  device: GPUDevice,
  batches: Array<{
    pipeline: GPUComputePipeline;
    bindGroup: GPUBindGroup;
    workgroups: number | [number, number, number];
    label?: string;
  }>,
  label?: string
): void;
