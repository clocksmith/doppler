import { getDevice } from '../../src/gpu/device.js';
import { getPipelineProvenance } from '../../src/gpu/kernels/pipeline-cache.js';

// Observe the pipeline bound at dispatch, including native encoder calls used
// by run and record adapters. Keep native receivers and parameters unchanged.
export async function observeKernelDispatches(operation) {
  const device = getDevice();
  const descriptor = Object.getOwnPropertyDescriptor(device, 'createCommandEncoder');
  const createCommandEncoder = device.createCommandEncoder;
  const dispatches = [];
  device.createCommandEncoder = (...args) => {
    const encoder = createCommandEncoder.apply(device, args);
    return new Proxy(encoder, {
      get(target, key) {
        if (key === 'beginComputePass') return (...passArgs) => {
          const pass = target.beginComputePass(...passArgs);
          let pipeline = null;
          return new Proxy(pass, {
            get(passTarget, passKey) {
              if (passKey === 'setPipeline') return value => {
                pipeline = value;
                return passTarget.setPipeline(value);
              };
              if (passKey === 'dispatchWorkgroups') return (...workgroups) => {
                dispatches.push({ workgroups, pipeline: getPipelineProvenance(pipeline) });
                return passTarget.dispatchWorkgroups(...workgroups);
              };
              const value = Reflect.get(passTarget, passKey, passTarget);
              return typeof value === 'function' ? value.bind(passTarget) : value;
            },
          });
        };
        const value = Reflect.get(target, key, target);
        return typeof value === 'function' ? value.bind(target) : value;
      },
    });
  };
  try {
    return { result: await operation(), dispatches };
  } finally {
    if (descriptor) Object.defineProperty(device, 'createCommandEncoder', descriptor);
    else delete device.createCommandEncoder;
  }
}
