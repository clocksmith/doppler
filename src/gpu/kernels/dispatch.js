


import { getDevice } from '../device.js';

function normalizeWorkgroups(workgroups) {
  if (typeof workgroups === 'number') {
    return [workgroups, 1, 1];
  }
  if (!Array.isArray(workgroups) || workgroups.length === 0) {
    throw new Error('dispatch requires workgroups as a number or [x, y, z]');
  }
  const x = workgroups[0] ?? 1;
  const y = workgroups[1] ?? 1;
  const z = workgroups[2] ?? 1;
  return [x, y, z];
}

function assertWorkgroupLimits(device, workgroups, label) {
  const maxPerDim = device?.limits?.maxComputeWorkgroupsPerDimension;
  const limit = Number.isFinite(maxPerDim) && maxPerDim > 0 ? maxPerDim : 65535;
  const [x, y, z] = normalizeWorkgroups(workgroups);
  if (x > limit || y > limit || z > limit) {
    throw new Error(
      `${label} dispatch exceeds maxComputeWorkgroupsPerDimension=${limit}: ` +
      `[${x}, ${y}, ${z}]`
    );
  }
  return [x, y, z];
}

export function dispatch(
  device,
  pipeline,
  bindGroup,
  workgroups,
  label = 'compute'
) {
  const [x, y, z] = assertWorkgroupLimits(device, workgroups, label);
  const encoder = device.createCommandEncoder({ label: `${label}_encoder` });
  const pass = encoder.beginComputePass({ label: `${label}_pass` });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(x, y, z);

  pass.end();
  device.queue.submit([encoder.finish()]);
}


export function dispatchKernel(
  target, // device or recorder
  pipeline,
  bindGroup,
  workgroups,
  label = 'compute'
) {
  if (target && typeof target.beginComputePass === 'function') {
    // Recorder
    recordDispatch(target, pipeline, bindGroup, workgroups, label);
  } else {
    // Device (or null if it should use default)
    const device = target || getDevice();
    dispatch(device, pipeline, bindGroup, workgroups, label);
  }
}

export function recordDispatch(
  recorder,
  pipeline,
  bindGroup,
  workgroups,
  label = 'compute'
) {
  const [x, y, z] = assertWorkgroupLimits(recorder.device, workgroups, label);
  const pass = recorder.beginComputePass(label);
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(x, y, z);

  pass.end();
}


export function dispatchIndirect(
  device,
  pipeline,
  bindGroup,
  indirectBuffer,
  indirectOffset = 0,
  label = 'compute'
) {
  const encoder = device.createCommandEncoder({ label: `${label}_encoder` });
  const pass = encoder.beginComputePass({ label: `${label}_pass` });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroupsIndirect(indirectBuffer, indirectOffset);
  pass.end();
  device.queue.submit([encoder.finish()]);
}


export function recordDispatchIndirect(
  recorder,
  pipeline,
  bindGroup,
  indirectBuffer,
  indirectOffset = 0,
  label = 'compute'
) {
  const pass = recorder.beginComputePass(label);
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroupsIndirect(indirectBuffer, indirectOffset);
  pass.end();
}


export function dispatchMultiBindGroup(
  device,
  pipeline,
  bindGroups,
  workgroups,
  label = 'compute'
) {
  const [x, y, z] = assertWorkgroupLimits(device, workgroups, label);
  const encoder = device.createCommandEncoder({ label: `${label}_encoder` });
  const pass = encoder.beginComputePass({ label: `${label}_pass` });
  pass.setPipeline(pipeline);

  for (let i = 0; i < bindGroups.length; i++) {
    pass.setBindGroup(i, bindGroups[i]);
  }

  pass.dispatchWorkgroups(x, y, z);

  pass.end();
  device.queue.submit([encoder.finish()]);
}


export function calculateWorkgroups1D(
  totalThreads,
  workgroupSize = 256
) {
  return Math.ceil(totalThreads / workgroupSize);
}


export function calculateWorkgroups2D(
  width,
  height,
  tileSize = 16
) {
  return [
    Math.ceil(width / tileSize),
    Math.ceil(height / tileSize),
  ];
}


export function calculateWorkgroups3D(
  width,
  height,
  depth,
  tileSizeX = 16,
  tileSizeY = 16,
  tileSizeZ = 1
) {
  return [
    Math.ceil(width / tileSizeX),
    Math.ceil(height / tileSizeY),
    Math.ceil(depth / tileSizeZ),
  ];
}


export function dispatchAdvanced(
  device,
  pipeline,
  workgroups,
  options = {}
) {
  const {
    label = 'compute',
    bindGroups = [],
    timestampWrites,
  } = options;

  const [x, y, z] = assertWorkgroupLimits(device, workgroups, label);
  const encoder = device.createCommandEncoder({ label: `${label}_encoder` });
  
  const passDescriptor = {
    label: `${label}_pass`,
  };

  if (timestampWrites) {
    passDescriptor.timestampWrites = timestampWrites;
  }

  const pass = encoder.beginComputePass(passDescriptor);
  pass.setPipeline(pipeline);

  // Set bind groups
  for (let i = 0; i < bindGroups.length; i++) {
    pass.setBindGroup(i, bindGroups[i]);
  }

  // Dispatch
  pass.dispatchWorkgroups(x, y, z);

  pass.end();
  device.queue.submit([encoder.finish()]);
}


export function dispatchBatch(
  device,
  batches,
  label = 'batch'
) {
  const encoder = device.createCommandEncoder({ label: `${label}_encoder` });

  for (const batch of batches) {
    const [x, y, z] = assertWorkgroupLimits(device, batch.workgroups, batch.label || label);
    const pass = encoder.beginComputePass({ label: batch.label || `${label}_pass` });
    pass.setPipeline(batch.pipeline);
    pass.setBindGroup(0, batch.bindGroup);
    pass.dispatchWorkgroups(x, y, z);

    pass.end();
  }

  device.queue.submit([encoder.finish()]);
}
