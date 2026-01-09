

import { dispatch, recordDispatch } from './dispatch.js';
import { getPipelineFast } from './utils.js';


export class KernelBase {
  
  device;

  
  constructor(device) {
    this.device = device;
  }

  
  async getPipelineFor(
    operation,
    variant,
    bindGroupLayout = null
  ) {
    return getPipelineFast(operation, variant, bindGroupLayout);
  }

  
  dispatchKernel(
    pipeline,
    bindGroup,
    workgroups,
    label
  ) {
    dispatch(this.device, pipeline, bindGroup, workgroups, label);
  }

  
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
