/**
 * Kernel Base - Shared dispatch and pipeline helpers for kernel wrappers.
 */

import { dispatch, recordDispatch } from './dispatch.js';
import { getPipelineFast } from './utils.js';

/**
 * @abstract
 */
export class KernelBase {
  /** @type {GPUDevice} */
  device;

  /**
   * @param {GPUDevice} device
   */
  constructor(device) {
    this.device = device;
  }

  /**
   * @protected
   * @param {string} operation
   * @param {string} variant
   * @param {GPUBindGroupLayout | null} [bindGroupLayout]
   * @returns {Promise<GPUComputePipeline>}
   */
  async getPipelineFor(
    operation,
    variant,
    bindGroupLayout = null
  ) {
    return getPipelineFast(operation, variant, bindGroupLayout);
  }

  /**
   * @protected
   * @param {GPUComputePipeline} pipeline
   * @param {GPUBindGroup} bindGroup
   * @param {number | [number, number, number]} workgroups
   * @param {string} label
   * @returns {void}
   */
  dispatchKernel(
    pipeline,
    bindGroup,
    workgroups,
    label
  ) {
    dispatch(this.device, pipeline, bindGroup, workgroups, label);
  }

  /**
   * @protected
   * @param {import('../command-recorder.js').CommandRecorder} recorder
   * @param {GPUComputePipeline} pipeline
   * @param {GPUBindGroup} bindGroup
   * @param {number | [number, number, number]} workgroups
   * @param {string} label
   * @returns {void}
   */
  recordKernel(
    recorder,
    pipeline,
    bindGroup,
    workgroups,
    label
  ) {
    recordDispatch(recorder, pipeline, bindGroup, workgroups, label);
  }
}
