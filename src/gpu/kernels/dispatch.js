


export function dispatch(
  device,
  pipeline,
  bindGroup,
  workgroups,
  label = 'compute'
) {
  const encoder = device.createCommandEncoder({ label: `${label}_encoder` });
  const pass = encoder.beginComputePass({ label: `${label}_pass` });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);

  if (typeof workgroups === 'number') {
    pass.dispatchWorkgroups(workgroups);
  } else {
    pass.dispatchWorkgroups(workgroups[0], workgroups[1], workgroups[2]);
  }

  pass.end();
  device.queue.submit([encoder.finish()]);
}


export function recordDispatch(
  recorder,
  pipeline,
  bindGroup,
  workgroups,
  label = 'compute'
) {
  const pass = recorder.beginComputePass(label);
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);

  if (typeof workgroups === 'number') {
    pass.dispatchWorkgroups(workgroups);
  } else {
    pass.dispatchWorkgroups(workgroups[0], workgroups[1], workgroups[2]);
  }

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
  const encoder = device.createCommandEncoder({ label: `${label}_encoder` });
  const pass = encoder.beginComputePass({ label: `${label}_pass` });
  pass.setPipeline(pipeline);

  for (let i = 0; i < bindGroups.length; i++) {
    pass.setBindGroup(i, bindGroups[i]);
  }

  if (typeof workgroups === 'number') {
    pass.dispatchWorkgroups(workgroups);
  } else {
    pass.dispatchWorkgroups(workgroups[0], workgroups[1], workgroups[2]);
  }

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
  if (typeof workgroups === 'number') {
    pass.dispatchWorkgroups(workgroups);
  } else {
    pass.dispatchWorkgroups(workgroups[0], workgroups[1], workgroups[2]);
  }

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
    const pass = encoder.beginComputePass({ label: batch.label || `${label}_pass` });
    pass.setPipeline(batch.pipeline);
    pass.setBindGroup(0, batch.bindGroup);

    if (typeof batch.workgroups === 'number') {
      pass.dispatchWorkgroups(batch.workgroups);
    } else {
      pass.dispatchWorkgroups(batch.workgroups[0], batch.workgroups[1], batch.workgroups[2]);
    }

    pass.end();
  }

  device.queue.submit([encoder.finish()]);
}
