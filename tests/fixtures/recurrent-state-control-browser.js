/** Browser-only diagnostic runner; retains GPU state between one-token dispatches. */
export async function runRecurrentStateControl({ input, expected, shader, mode }) {
  if (!['reference-reset', 'continuous'].includes(mode)) throw Error('Unknown recurrent state control');
  const p = input.params;
  const decode = data => {
    const text = atob(data), bytes = new Uint8Array(text.length);
    for (let i = 0; i < text.length; i++) bytes[i] = text.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  };
  const tensors = Object.fromEntries(input.tensors.map(t => [t.role, decode(t.data)]));
  tensors.convOutput = decode(expected.tensors.filter(t => t.role === 'convOutput')[0].data);
  const referenceStates = mode === 'reference-reset' ? decode(globalThis.referenceStateChunks.join('')) : null;
  delete globalThis.referenceStateChunks;
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw Error('Recurrent state control requires a physical WebGPU adapter');
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 9 } });
  const owned = []; let allocatedBytes = 0;
  const buffer = (size, usage) => {
    const b = device.createBuffer({ size, usage }); owned.push(b); allocatedBytes += size; return b;
  };
  try {
    const module = device.createShaderModule({ code: shader });
    const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error');
    if (errors.length) throw Error(errors.map(e => e.message).join('\n'));
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: {
      module, entryPoint: 'main', constants: { WORKGROUP_SIZE: 128 } } });
    const uniformBytes = new ArrayBuffer(64), view = new DataView(uniformBytes);
    ['numTokens', 'convDim', 'convKernelSize', 'numVHeads', 'numKHeads', 'headKDim',
      'headVDim', 'qSize', 'kSize', 'valueDim', 'qRep'].forEach((key, i) => view.setUint32(i * 4, key === 'numTokens' ? 1 : p[key], true));
    view.setUint32(44, Number(p.normMode === 'per_head'), true);
    view.setFloat32(48, p.rmsNormEps, true); view.setFloat32(52, p.qkL2NormEps, true);
    view.setUint32(56, Number(p.abPacked) | Number(p.qkvzPacked) << 1, true);
    view.setUint32(60, p.bProjOffsetElements, true);
    const uniform = buffer(64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(uniform, 0, uniformBytes);
    const stateElements = p.numVHeads * p.headKDim * p.headVDim;
    const zStride = p.valueDim + (p.qkvzPacked ? p.convDim : 0);
    const sizes = { convOutput: p.convDim, z: zStride, a: p.numVHeads,
      b: p.numVHeads + (p.abPacked ? p.bProjOffsetElements : 0), dtBias: tensors.dtBias.length,
      aLog: tensors.aLog.length, normWeight: tensors.normWeight.length, recurrentState: stateElements, output: p.valueDim };
    const roles = Object.keys(sizes);
    const gpu = Object.fromEntries(roles.map(role => [role,
      buffer(sizes[role] * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST)]));
    for (const role of ['dtBias', 'aLog', 'normWeight']) device.queue.writeBuffer(gpu[role], 0, tensors[role]);
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries:
      [uniform, ...roles.map(role => gpu[role])].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const readbackState = buffer(stateElements * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    const readbackOutput = buffer(p.valueDim * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    const read = async b => {
      await b.mapAsync(GPUMapMode.READ);
      const bytes = new Uint8Array(b.getMappedRange().slice(0)); b.unmap();
      let text = '';
      for (let i = 0; i < bytes.length; i += 16384) text += String.fromCharCode(...bytes.subarray(i, i + 16384));
      return btoa(text);
    };
    const tokens = []; let stateWrites = 0;
    device.pushErrorScope('validation');
    for (let token = 0; token < p.numTokens; token++) {
      for (const [role, stride] of [['convOutput', p.convDim], ['z', zStride], ['a', p.numVHeads]]) {
        device.queue.writeBuffer(gpu[role], 0, tensors[role].subarray(token * stride, (token + 1) * stride));
      }
      const bOffset = p.abPacked ? p.bProjOffsetElements : 0;
      device.queue.writeBuffer(gpu.b, bOffset * 4, tensors.b.subarray(bOffset + token * p.numVHeads, bOffset + (token + 1) * p.numVHeads));
      if (mode === 'reference-reset' || token === 0) {
        const state = token === 0 ? tensors.recurrentState.subarray(0, stateElements)
          : referenceStates.subarray((token - 1) * stateElements, token * stateElements);
        device.queue.writeBuffer(gpu.recurrentState, 0, state); stateWrites++;
      }
      const started = performance.now();
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(p.numVHeads); pass.end();
      encoder.copyBufferToBuffer(gpu.recurrentState, 0, readbackState, 0, stateElements * 4);
      encoder.copyBufferToBuffer(gpu.output, 0, readbackOutput, 0, p.valueDim * 4);
      device.queue.submit([encoder.finish()]);
      tokens.push({ token, output: await read(readbackOutput), state: await read(readbackState),
        submitAndReadbackMs: performance.now() - started });
    }
    const validationError = await device.popErrorScope();
    if (validationError) throw Error(validationError.message);
    return { tokens, stateWrites, allocatedBytes, vendor: adapter.info.vendor, architecture: adapter.info.architecture };
  } finally { for (const b of owned) b.destroy(); device.destroy(); }
}
