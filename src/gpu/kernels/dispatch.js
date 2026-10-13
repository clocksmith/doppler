


import { getDevice, getLastDeviceLossInfo } from '../device.js';

// Explicit diagnostic leases require exclusive dispatch ownership of a device.
// They observe immediate submissions without batching or changing uniforms.
const immediateProfiles = new WeakMap();

function requireProfileRecorder(device, getRecorder) {
  const recorder = getRecorder();
  if (recorder?.device !== device) throw new Error('Immediate profile recorder device mismatch.');
  if (!recorder.isProfilingEnabled()) throw new Error('Immediate profiling requires working timestamp queries.');
  return recorder;
}

export function beginImmediateDispatchProfile(device, getRecorder) {
  if (immediateProfiles.has(device)) throw new Error('Immediate profiling device is already leased.');
  requireProfileRecorder(device, getRecorder);
  const lease = { getRecorder };
  immediateProfiles.set(device, lease);
  return () => {
    if (immediateProfiles.get(device) === lease) immediateProfiles.delete(device);
  };
}

function requireDispatchDevice(device, label) {
  if (device) return device;
  const loss = getLastDeviceLossInfo();
  throw new Error(
    `${label} dispatch requires an active GPU device; ` +
    `lastDeviceLoss=${JSON.stringify(loss)}`
  );
}

function normalizeWorkgroups(workgroups) {
  if (typeof workgroups === 'number') {
    return [workgroups, 1, 1];
  }
  if (!Array.isArray(workgroups) || workgroups.length === 0) {
    throw new Error('dispatch requires workgroups as a number or [x, y, z]');
  }
  if (workgroups.length >= 3 && workgroups[1] !== undefined && workgroups[2] !== undefined) {
    return workgroups;
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
  requireDispatchDevice(device, label);
  const [x, y, z] = assertWorkgroupLimits(device, workgroups, label);
  const profile = immediateProfiles.get(device);
  if (profile) {
    const recorder = requireProfileRecorder(device, profile.getRecorder);
    recorder.recordDispatch(pipeline, bindGroup, [x, y, z], `${label}:${pipeline.label} [${x},${y},${z}]`);
    recorder.submitPartial();
    return;
  }
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
  const dispatchWorkgroups = assertWorkgroupLimits(recorder.device, workgroups, label);
  if (typeof recorder.recordDispatch === 'function') {
    recorder.recordDispatch(pipeline, bindGroup, dispatchWorkgroups, label);
    return;
  }
  const [x, y, z] = dispatchWorkgroups;
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
  requireDispatchDevice(device, label);
  const profile = immediateProfiles.get(device);
  if (profile) {
    const recorder = requireProfileRecorder(device, profile.getRecorder);
    recorder.recordDispatchIndirect(pipeline, bindGroup, indirectBuffer, indirectOffset, `${label}:${pipeline.label} [indirect]`);
    recorder.submitPartial();
    return;
  }
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
  if (typeof recorder.recordDispatchIndirect === 'function') {
    recorder.recordDispatchIndirect(pipeline, bindGroup, indirectBuffer, indirectOffset, label);
    return;
  }
  const pass = recorder.beginComputePass(label);
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroupsIndirect(indirectBuffer, indirectOffset);
  pass.end();
}
