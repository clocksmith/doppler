/**
 * Kernel Base - Shared dispatch and pipeline helpers for kernel wrappers.
 */

import type { CommandRecorder } from '../command-recorder.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { createPipeline } from './utils.js';

export abstract class KernelBase {
  protected readonly device: GPUDevice;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  protected async getPipelineFor(
    operation: string,
    variant: string,
    bindGroupLayout: GPUBindGroupLayout | null = null
  ): Promise<GPUComputePipeline> {
    return createPipeline(operation, variant, bindGroupLayout);
  }

  protected dispatchKernel(
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    workgroups: number | [number, number, number],
    label: string
  ): void {
    dispatch(this.device, pipeline, bindGroup, workgroups, label);
  }

  protected recordKernel(
    recorder: CommandRecorder,
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    workgroups: number | [number, number, number],
    label: string
  ): void {
    recordDispatch(recorder, pipeline, bindGroup, workgroups, label);
  }
}
