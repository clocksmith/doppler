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

