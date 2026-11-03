import { readbackBuffer } from '../../src/gpu/readback-buffer.js';

// Portable fixture shared by Node and physical-browser diagnostics. The serial
// convolution is an independent GPU control for every output bit; CPU samples
// and final history check the causal window and nonzero incoming state.
export async function checkConvGridReference(device, sources, shape) {
  const { numTokens, convDim, window, packedValueDim = 0 } = shape;
  const width = sources.workgroupSize;
  const stride = convDim + packedValueDim;
  const qkv = new Float32Array(numTokens * stride);
  for (let i = 0; i < qkv.length; i++) qkv[i] = ((i % 127) - 63) / 1280;
  const weights = Float32Array.from({ length: convDim * window }, (_, i) => ((i % 31) - 15) / 128);
  const initial = Float32Array.from({ length: convDim * window }, (_, i) => ((i % 43) - 21) / 256);
  const buffers = [];
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const upload = (data, uniform = false) => {
    const buffer = device.createBuffer({ size: data.byteLength,
      usage: uniform ? GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST : usage });
    buffers.push(buffer); device.queue.writeBuffer(buffer, 0, data); return buffer;
  };
  const params = new Uint32Array(16);
  params[0] = numTokens; params[1] = convDim; params[2] = window;
  params[9] = packedValueDim; params[14] = packedValueDim ? 2 : 0;
  let scopeOpen = false;
  try {
    const common = [upload(params, true), upload(qkv), upload(weights)];
    const serialModule = device.createShaderModule({ code: sources.serial });
    const parallelModule = device.createShaderModule({ code: sources.parallel });
    const [serial, prefix, parallel] = await Promise.all([
      [serialModule, 'main'], [parallelModule, 'main_prefix'], [parallelModule, 'main_parallel'],
    ].map(([module, entryPoint]) => device.createComputePipelineAsync({ layout: 'auto',
      compute: { module, entryPoint, constants: { WORKGROUP_SIZE: width } } })));
    const controls = [];
    device.pushErrorScope('validation'); scopeOpen = true;
    for (const split of [false, true]) {
      const state = upload(initial);
      const output = device.createBuffer({ size: numTokens * convDim * 4, usage });
      buffers.push(output);
      const resources = [...common, state, output];
      const encoder = device.createCommandEncoder();
      for (const pipeline of split ? [prefix, parallel] : [serial]) {
        if (pipeline === parallel && numTokens < window) continue;
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
          entries: resources.map((buffer, binding) => ({ binding, resource: { buffer } })) }));
        pass.dispatchWorkgroups(Math.ceil(convDim / width), pipeline === parallel ? numTokens : 1, 1);
        pass.end();
      }
      device.queue.submit([encoder.finish()]);
      controls.push({ output, state });
    }
    await device.queue.onSubmittedWorkDone();
    scopeOpen = false;
    const error = await device.popErrorScope();
    if (error) throw new Error(error.message);
    const [baseline, candidate] = await Promise.all(controls.map(async ({ output, state }) => ({
      output: new Uint32Array(await readbackBuffer(device, output, numTokens * convDim * 4)),
      state: new Float32Array(await readbackBuffer(device, state, initial.byteLength)),
    })));
    for (let i = 0; i < baseline.output.length; i++) {
      if (candidate.output[i] !== baseline.output[i]) throw new Error(`Convolution bit mismatch at ${i}`);
    }
    for (let channel = 0; channel < convDim; channel++) {
      for (let k = 0; k < window; k++) {
        const token = numTokens - window + k;
        const expected = token < 0 ? initial[channel * window + window + token] : qkv[token * stride + channel];
        const at = channel * window + k;
        if (candidate.state[at] !== expected || baseline.state[at] !== expected) {
          throw new Error(`Convolution history mismatch at ${at}`);
        }
      }
    }
    const values = new Float32Array(candidate.output.buffer);
    let maxCpuSampleError = 0;
    for (const token of new Set([0, Math.min(window - 1, numTokens - 1), Math.floor(numTokens / 2), numTokens - 1])) {
      for (const channel of new Set([0, Math.min(width - 1, convDim - 1), Math.min(width, convDim - 1), convDim - 1])) {
        let sum = 0;
        for (let k = 0; k < window; k++) {
          const sourceToken = token + 1 - window + k;
          const value = sourceToken < 0 ? initial[channel * window + window + sourceToken] : qkv[sourceToken * stride + channel];
          sum += value * weights[channel * window + k];
        }
        const expected = sum / (1 + Math.exp(-sum));
        const delta = Math.abs(values[token * convDim + channel] - expected);
        if (!Number.isFinite(delta) || delta > 1e-6) throw new Error(`CPU convolution mismatch at ${token}, ${channel}`);
        maxCpuSampleError = Math.max(maxCpuSampleError, delta);
      }
    }
    return { ...shape, exactOutputBits: baseline.output.length, exactHistory: initial.length,
      maxCpuSampleError, parallelGrid: [Math.ceil(convDim / width), numTokens, 1] };
  } finally {
    if (scopeOpen) await device.popErrorScope();
    await device.queue.onSubmittedWorkDone().catch(() => {});
    for (const buffer of buffers) buffer.destroy();
  }
}
