
export async function probeSubmitLatency(device) {
  if (!device) return null;

  let gpuBuffer = null;
  let stagingBuffer = null;

  try {
    gpuBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    stagingBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const { getShaderModule } = await import('./kernels/shader-cache.js');
    const module = await getShaderModule(device, 'submit_probe.wgsl', 'submit_probe');
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: gpuBuffer } }],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyBufferToBuffer(gpuBuffer, 0, stagingBuffer, 0, 4);

    const start = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    stagingBuffer.unmap();
    const elapsed = performance.now() - start;

    return elapsed;
  } catch {
    return null;
  } finally {
    stagingBuffer?.destroy();
    gpuBuffer?.destroy();
  }
}
