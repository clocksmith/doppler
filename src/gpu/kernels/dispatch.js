/**
 * Dispatch Helpers - Simplified GPU kernel dispatch
 *
 * Provides helpers to reduce boilerplate for common dispatch patterns:
 * - Single submit dispatch
 * - CommandRecorder dispatch (batched)
 * - Multi-dimensional dispatch
 */

/**
 * Dispatch a single compute pass and submit immediately
 * Use for standalone kernels that don't participate in batching
 * @param {GPUDevice} device
 * @param {GPUComputePipeline} pipeline
 * @param {GPUBindGroup} bindGroup
 * @param {number | [number, number, number]} workgroups
 * @param {string} [label]
 * @returns {void}
 */
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

/**
 * Record a compute pass to a CommandRecorder (no submit)
 * Use for kernels in the batched pipeline path
 * @param {import('../command-recorder.js').CommandRecorder} recorder
 * @param {GPUComputePipeline} pipeline
 * @param {GPUBindGroup} bindGroup
 * @param {number | [number, number, number]} workgroups
 * @param {string} [label]
 * @returns {void}
 */
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

/**
 * Dispatch a single compute pass using an indirect dispatch buffer
 * Use when workgroup counts are produced on GPU
 * @param {GPUDevice} device
 * @param {GPUComputePipeline} pipeline
 * @param {GPUBindGroup} bindGroup
 * @param {GPUBuffer} indirectBuffer
 * @param {number} [indirectOffset]
 * @param {string} [label]
 * @returns {void}
 */
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

/**
 * Record an indirect dispatch into a CommandRecorder (no submit)
 * @param {import('../command-recorder.js').CommandRecorder} recorder
 * @param {GPUComputePipeline} pipeline
 * @param {GPUBindGroup} bindGroup
 * @param {GPUBuffer} indirectBuffer
 * @param {number} [indirectOffset]
 * @param {string} [label]
 * @returns {void}
 */
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

/**
 * Dispatch with multiple bind groups
 * For kernels that use multiple bind group sets
 * @param {GPUDevice} device
 * @param {GPUComputePipeline} pipeline
 * @param {GPUBindGroup[]} bindGroups
 * @param {number | [number, number, number]} workgroups
 * @param {string} [label]
 * @returns {void}
 */
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

/**
 * Calculate workgroup count for 1D dispatch
 * @param {number} totalThreads - Total number of threads needed
 * @param {number} [workgroupSize] - Threads per workgroup (default: 256)
 * @returns {number} Number of workgroups (rounded up)
 */
export function calculateWorkgroups1D(
  totalThreads,
  workgroupSize = 256
) {
  return Math.ceil(totalThreads / workgroupSize);
}

/**
 * Calculate workgroup count for 2D dispatch
 * @param {number} width - Width dimension (e.g., matrix columns)
 * @param {number} height - Height dimension (e.g., matrix rows)
 * @param {number} [tileSize] - Tile size per workgroup (default: 16)
 * @returns {[number, number]} [workgroupsX, workgroupsY]
 */
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

/**
 * Calculate workgroup count for 3D dispatch
 * @param {number} width - Width dimension
 * @param {number} height - Height dimension
 * @param {number} depth - Depth dimension
 * @param {number} [tileSizeX] - Tile size in X (default: 16)
 * @param {number} [tileSizeY] - Tile size in Y (default: 16)
 * @param {number} [tileSizeZ] - Tile size in Z (default: 1)
 * @returns {[number, number, number]} [workgroupsX, workgroupsY, workgroupsZ]
 */
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

/**
 * Advanced dispatch with full control
 * Supports push constants, timestamps, and multiple bind groups
 * @param {GPUDevice} device
 * @param {GPUComputePipeline} pipeline
 * @param {number | [number, number, number]} workgroups
 * @param {import('./dispatch.js').DispatchOptions} [options]
 * @returns {void}
 */
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
  /** @type {GPUComputePassDescriptor} */
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

/**
 * Batch multiple dispatches in a single command buffer
 * Useful for multi-kernel operations that should be submitted together
 * @param {GPUDevice} device
 * @param {Array<{ pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, workgroups: number | [number, number, number], label?: string }>} batches
 * @param {string} [label]
 * @returns {void}
 */
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
