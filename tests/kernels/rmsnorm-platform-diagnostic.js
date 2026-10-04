/** Isolated numerical experiment. Candidate WGSL never enters model execution. */
export async function diagnoseRMSNorm({ source, input, weights, epsilon, weightOffset }) {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw Error('WebGPU adapter required');
  const device = await adapter.requestDevice({ requiredFeatures: ['subgroups'] });
  const size = input.length, results = [];
  const candidate = source.replace('let inv_rms = 1.0 / rms;', `let a = mean_sq + u.eps;
    let root_error = fma(-rms, rms, a);
    let refined_root = rms + root_error / (2.0 * rms);
    let reciprocal = 1.0 / refined_root;
    let reciprocal_error = fma(-refined_root, reciprocal, 1.0);
    let inv_rms = fma(reciprocal, reciprocal_error, reciprocal);`);
  if (candidate === source) throw Error('Canonical normalization boundary not found');
  try {
    for (const [variant, code] of [['canonical', source], ['refined-f32-diagnostic', candidate]]) {
      const owned = [];
      const buffer = (bytes, usage) => { const value = device.createBuffer({ size: bytes, usage }); owned.push(value); return value; };
      try {
        const shader = device.createShaderModule({ code });
        const errors = (await shader.getCompilationInfo()).messages.filter(message => message.type === 'error');
        if (errors.length) throw Error(errors.map(error => error.message).join('\n'));
        const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: shader,
          entryPoint: 'main', constants: { WORKGROUP_SIZE: 256, RMS_NORM_OFFSET: weightOffset, WEIGHT_IS_F16: false } } });
        const uniform = buffer(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        const data = new ArrayBuffer(32), view = new DataView(data);
        view.setUint32(0, size, true); view.setUint32(4, 1, true); view.setFloat32(8, epsilon, true);
        view.setUint32(16, 1, true); view.setFloat32(20, 1, true); device.queue.writeBuffer(uniform, 0, data);
        const x = buffer(size * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        const w = buffer(size * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        const y = buffer(size * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        const residual = buffer(size * 4, GPUBufferUsage.STORAGE);
        const prenorm = buffer(size * 4, GPUBufferUsage.STORAGE);
        const readback = buffer(size * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
        device.queue.writeBuffer(x, 0, new Float32Array(input)); device.queue.writeBuffer(w, 0, new Float32Array(weights));
        const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries:
          [uniform, x, w, y, residual, prenorm].map((buffer, binding) => ({ binding, resource: { buffer } })) });
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
        encoder.copyBufferToBuffer(y, 0, readback, 0, size * 4); device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const values = Array.from(new Float32Array(readback.getMappedRange().slice(0))); readback.unmap();
        results.push({ variant, values });
      } finally { for (const value of owned) value.destroy(); }
    }
    return { adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture, description: adapter.info.description }, results };
  } finally { device.destroy(); }
}
