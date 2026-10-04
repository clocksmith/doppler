/** Physical wrapper diagnostic. Captures descriptors and bytes without replacing compute. */
export async function auditRMSNormReuse({ moduleBase, input, weights, graph }) {
  const load = path => import(new URL(path, moduleBase));
  const { initDevice } = await load('gpu/device.js');
  const { createTensor } = await load('gpu/tensor.js');
  const { createWeightBuffer } = await load('gpu/weight-buffer.js');
  const { runRMSNorm, recordRMSNorm, selectRMSNormKernel } = await load('gpu/kernels/rmsnorm.js');
  const { getCachedPipeline } = await load('gpu/kernels/pipeline-cache.js');
  const { getKernelConfig } = await load('gpu/kernels/kernel-configs.js');
  const { CommandRecorder } = await load('gpu/command-recorder.js');
  const { releaseBuffer } = await load('memory/buffer-pool.js');
  const device = await initDevice();
  const owned = [], pipelines = [], calls = [], errors = [], restore = [];
  const modules = new WeakMap(), identities = new WeakMap(), uniforms = new WeakMap();
  const groups = new WeakMap(), passes = new WeakMap();
  let active = null, nextBuffer = 0;
  const buffers = new WeakMap();
  const identify = buffer => {
    if (!buffers.has(buffer)) buffers.set(buffer, nextBuffer++);
    return { id: buffers.get(buffer), bytes: buffer.size, label: buffer.label };
  };
  const patch = (object, name, wrap) => {
    const original = object[name]; object[name] = wrap(original);
    restore.push(() => { object[name] = original; });
  };
  device.addEventListener('uncapturederror', event => errors.push(event.error.message));
  patch(GPUDevice.prototype, 'createShaderModule', original => function (descriptor) {
    const module = original.call(this, descriptor); modules.set(module, descriptor.code); return module;
  });
  for (const method of ['createComputePipeline', 'createComputePipelineAsync']) {
    patch(GPUDevice.prototype, method, original => function (descriptor) {
      const observe = pipeline => {
        if (descriptor.label?.startsWith('rmsnorm_')) {
          const record = { id: pipelines.length, entryPoint: descriptor.compute.entryPoint,
            constants: { ...descriptor.compute.constants }, layout: descriptor.layout,
            source: modules.get(descriptor.compute.module) };
          identities.set(pipeline, record.id); pipelines.push(record);
        }
        return pipeline;
      };
      const value = original.call(this, descriptor);
      return method.endsWith('Async') ? value.then(observe) : observe(value);
    });
  }
  patch(GPUQueue.prototype, 'writeBuffer', original => function (buffer, offset, data, dataOffset = 0, size) {
    if (buffer.label?.includes('rmsnorm') && buffer.label.includes('uniform')) {
      const scale = data.BYTES_PER_ELEMENT ?? 1;
      const source = ArrayBuffer.isView(data) ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : new Uint8Array(data);
      const bytes = uniforms.get(buffer) ?? new Uint8Array(buffer.size);
      bytes.set(source.slice(dataOffset * scale, size === undefined ? undefined : (dataOffset + size) * scale), offset);
      uniforms.set(buffer, bytes);
    }
    return original.call(this, buffer, offset, data, dataOffset, size);
  });
  patch(device, 'createBindGroup', original => function (descriptor) {
    const group = original.call(this, descriptor);
    if (descriptor.label === 'rmsnorm_bind_group') groups.set(group, descriptor.entries.map(entry => ({
      binding: entry.binding, buffer: identify(entry.resource.buffer), offset: entry.resource.offset ?? 0,
      size: entry.resource.size ?? entry.resource.buffer.size - (entry.resource.offset ?? 0),
      uniformBytes: entry.binding === 0 ? Array.from(uniforms.get(entry.resource.buffer) ?? []) : null,
    })));
    return group;
  });
  patch(GPUComputePassEncoder.prototype, 'setPipeline', original => function (pipeline) {
    const state = passes.get(this) ?? {}; state.pipeline = identities.get(pipeline); passes.set(this, state);
    return original.call(this, pipeline);
  });
  patch(GPUComputePassEncoder.prototype, 'setBindGroup', original => function (...args) {
    if (args[0] === 0) { const state = passes.get(this) ?? {}; state.group = groups.get(args[1]); passes.set(this, state); }
    return original.apply(this, args);
  });
  patch(GPUComputePassEncoder.prototype, 'dispatchWorkgroups', original => function (...dimensions) {
    if (active) active.dispatches.push({ ...passes.get(this), dimensions });
    return original.apply(this, dimensions);
  });
  const buffer = (data, label) => {
    const value = device.createBuffer({ label, size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    owned.push(value); device.queue.writeBuffer(value, 0, data); return value;
  };
  const digest = async bytes => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('');
  try {
    const x = buffer(new Float32Array(input), 'audit_input');
    const w = buffer(new Uint16Array(weights), 'audit_weight_f16');
    const residual = buffer(new Float32Array(input), 'audit_residual');
    const prenorm = buffer(new Float32Array(input.length), 'audit_prenorm');
    const tensor = createTensor(x, 'f32', [1, input.length], 'audit_input');
    const weight = createWeightBuffer(w, 'f16', 'row', [weights.length], 'audit_weight');
    const options = { batchSize: 1, hiddenSize: input.length, rmsNormWeightOffset: true,
      kernelPath: graph, role: 'input_norm', section: 'layer', phase: 'decode', layerIdx: 0 };
    const cases = [
      { name: 'initial', epsilon: 1e-6 }, { name: 'reuse', epsilon: 1e-6 },
      { name: 'changed_uniforms', epsilon: 2e-6, options: { outputScale: 0.5 } },
      { name: 'changed_constants', epsilon: 1e-6, options: { rmsNormWeightOffset: false } },
      { name: 'pre_residual', epsilon: 1e-6, options: { preResidual: residual, residualSumOutput: prenorm } },
      { name: 'original_again', epsilon: 1e-6 },
    ];
    for (const recorded of [false, true]) for (const test of cases) {
      const callOptions = { ...options, ...test.options };
      const variant = selectRMSNormKernel(callOptions, false), config = getKernelConfig('rmsnorm', variant);
      const constants = { RMS_NORM_OFFSET: callOptions.rmsNormWeightOffset, WEIGHT_IS_F16: true,
        PRE_RESIDUAL: !!callOptions.preResidual, OUTPUT_PRENORM: !!callOptions.residualSumOutput };
      const cached = getCachedPipeline('rmsnorm', variant, constants, device);
      const recorder = recorded ? new CommandRecorder(device) : null;
      active = { name: test.name, recorded, variant, graphEntry: 'main', epsilon: test.epsilon,
        outputScale: callOptions.outputScale ?? 1, input: identify(x), weight: identify(w),
        shape: tensor.shape, weightShape: weight.shape, weightLayout: weight.layout,
        requested: { shaderFile: config.shaderFile, entryPoint: config.entryPoint, constants },
        cachedPipelineId: cached ? identities.get(cached) : null, dispatches: [] };
      let result;
      try {
        result = recorded ? await recordRMSNorm(recorder, tensor, weight, test.epsilon, callOptions)
          : await runRMSNorm(tensor, weight, test.epsilon, callOptions);
        if (recorder) await recorder.submitAndWait();
        const staging = device.createBuffer({ size: input.length * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        try {
          const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(result.buffer, 0, staging, 0, staging.size);
          device.queue.submit([encoder.finish()]); await staging.mapAsync(GPUMapMode.READ);
          active.outputSha256 = await digest(staging.getMappedRange().slice(0)); staging.unmap();
        } finally { staging.destroy(); }
        calls.push(active);
      } finally { active = null; if (result) releaseBuffer(result.buffer); recorder?.abort(); }
    }
    for (const pipeline of pipelines) {
      pipeline.shaderSha256 = await digest(new TextEncoder().encode(pipeline.source)); delete pipeline.source;
    }
    await device.queue.onSubmittedWorkDone();
    return { pipelines, calls, errors };
  } finally {
    for (const undo of restore.reverse()) undo();
    for (const value of owned) value.destroy();
    device.destroy();
  }
}
