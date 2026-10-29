try {
  const config = await (await fetch('/probe-config.json')).json();
  const fixture = await (await fetch('/canonical/rne-fixture.json')).json();
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter || adapter.info.vendor.toLowerCase() !== config.requiredVendor) throw new Error('Required physical adapter unavailable');
  const device = await adapter.requestDevice({ requiredFeatures: ['shader-f16'] });
  const errors = [], checks = [];
  device.addEventListener('uncapturederror', event => errors.push(event.error.message));
  const sources = config.roundingShaders;
  if (!sources?.native || !sources?.prototype) throw new Error('Explicit rounding comparison shader paths are required');
  for (const [name, url] of Object.entries(sources)) {
    const shader = await (await fetch(url)).text();
    const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(shader)))].map(x => x.toString(16).padStart(2, '0')).join('');
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: {
      module: device.createShaderModule({ code: shader }), entryPoint: 'main', constants: { WORKGROUP_SIZE: config.contract.materializationObservation.workgroupSize } } });
    for (const range of fixture.ranges) {
      const owned = [];
      const make = (size, usage) => { const buffer = device.createBuffer({ size: Math.ceil(size / 4) * 4, usage }); owned.push(buffer); return buffer; };
      try {
        const input = new Uint32Array(fixture.bits), size = (range.dstOffset + range.count + 3) * 2;
        const initial = new Uint16Array(Math.ceil(size / 4) * 2).fill(0x3555);
        const expected = initial.slice();
        const expectedV = initial.slice();
        for (let i = 0; i < range.count; i++) expected[range.dstOffset + i] = fixture.expected[range.srcOffset + i];
        for (let i = 0; i < range.count; i++) expectedV[range.dstOffset + i] = fixture.expected[range.srcOffset + i] ^ 0x8000;
        const uniform = make(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        const inputBuffer = make(input.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        const inputV = make(input.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        const outputK = make(initial.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
        const outputV = make(initial.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
        const staging = make(initial.byteLength, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
        device.queue.writeBuffer(uniform, 0, new Uint32Array([range.srcOffset, range.dstOffset, range.count, 0]));
        device.queue.writeBuffer(inputBuffer, 0, input); device.queue.writeBuffer(outputK, 0, initial); device.queue.writeBuffer(outputV, 0, initial);
        device.queue.writeBuffer(inputV, 0, Uint32Array.from(input, value => value ^ 0x80000000));
        const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [uniform, inputBuffer, inputV, outputK, outputV].map((buffer, binding) => ({ binding, resource: { buffer } })) });
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(range.count / config.contract.materializationObservation.workgroupSize)); pass.end();
        encoder.copyBufferToBuffer(outputK, 0, staging, 0, initial.byteLength); device.queue.submit([encoder.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const actual = new Uint16Array(staging.getMappedRange().slice(0)); staging.unmap();
        let mismatches = 0;
        for (let i = 0; i < expected.length; i++) if (actual[i] !== expected[i]) mismatches++;
        const second = device.createCommandEncoder(); second.copyBufferToBuffer(outputV, 0, staging, 0, initial.byteLength); device.queue.submit([second.finish()]);
        await staging.mapAsync(GPUMapMode.READ); const actualV = new Uint16Array(staging.getMappedRange().slice(0)); staging.unmap();
        let valueMismatches = 0;
        for (let i = 0; i < expectedV.length; i++) if (actualV[i] !== expectedV[i]) valueMismatches++;
        checks.push({ name, shaderSha256: digest, ...range, elements: expected.length, mismatches, valueMismatches });
      } finally { for (const buffer of owned) buffer.destroy(); }
    }
  }
  if (errors.length) throw new Error(errors.join('\n'));
  globalThis.RESULT = { schema: 'doppler.kv-rounding-isolation.v1', checks, errors,
    pass: checks.filter(x => x.name === 'prototype').every(x => x.mismatches === 0 && x.valueMismatches === 0),
    scope: 'Independent Python struct binary16 oracle; no model or speed claim.' };
  device.destroy();
} catch (error) { globalThis.RESULT = { error: String(error.stack || error) }; console.error(error); }
