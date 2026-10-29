// Untimed observation only: read the native engine's actual storage buffers.
export function observePooledUploads(Engine, weights) {
  const sources = new Map(weights.materializationBindings.map(x => [x.value.data, x]));
  const uploaded = new Map(), original = Engine.prototype._buf;
  Engine.prototype._buf = function (data, usage) {
    const buffer = original.call(this, data, usage);
    const binding = sources.get(data);
    if (binding) uploaded.set(binding.record.name, buffer);
    return buffer;
  };
  return { uploaded, restore() { Engine.prototype._buf = original; sources.clear(); } };
}

export async function verifyPooledGpu(device, weights, uploaded, policy) {
  const chunkBytes = policy.chunkBytes, workgroupSize = policy.workgroupSize;
  if (!Number.isSafeInteger(chunkBytes) || chunkBytes <= 0 || chunkBytes % 4 ||
      !Number.isSafeInteger(workgroupSize) || workgroupSize <= 0 || workgroupSize > device.limits.maxComputeWorkgroupSizeX ||
      Math.ceil(chunkBytes / 4 / workgroupSize) > device.limits.maxComputeWorkgroupsPerDimension) {
    throw new Error('Invalid materialization observation geometry');
  }
  const module = device.createShaderModule({ code: `
    @group(0) @binding(0) var<storage,read> input: array<u32>;
    @group(0) @binding(1) var<storage,read_write> output: array<u32>;
    @group(0) @binding(2) var<uniform> extent: vec4<u32>;
    @compute @workgroup_size(${workgroupSize}) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
      if (id.x < extent.y) { output[id.x] = input[extent.x + id.x]; }
    }` });
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  const output = device.createBuffer({ size: chunkBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const staging = device.createBuffer({ size: chunkBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const extent = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const digest = async bytes => [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(x => x.toString(16).padStart(2, '0')).join('');
  const checks = [];
  try {
    for (const { record, value } of weights.materializationBindings) {
      let actual, location;
      if (value === weights.embed) {
        actual = await digest(value.data); location = 'native-cpu-embedding';
      } else {
        const entry = value.gpu?.buf ?? uploaded.get(record.name);
        if (!entry) throw new Error(`No actual GPU resource for ${record.name}`);
        const buffer = entry.__view ? entry.buffer : entry;
        const offset = entry.__view ? entry.offset : 0;
        const bytes = record.shape.reduce((a, b) => a * b, 1) * 4;
        if (offset % 4 || offset + bytes > buffer.size || !(buffer.usage & GPUBufferUsage.STORAGE)) {
          throw new Error(`Invalid native weight extent: ${record.name}`);
        }
        const observed = new Uint8Array(bytes);
        const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [buffer, output, extent].map((b, binding) => ({ binding, resource: { buffer: b } })) });
        for (let position = 0; position < bytes; position += chunkBytes) {
          const count = Math.min(chunkBytes, bytes - position);
          device.queue.writeBuffer(extent, 0, new Uint32Array([(offset + position) / 4, count / 4, 0, 0]));
          const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
          pass.setPipeline(pipeline); pass.setBindGroup(0, group);
          pass.dispatchWorkgroups(Math.ceil(count / 4 / workgroupSize)); pass.end();
          encoder.copyBufferToBuffer(output, 0, staging, 0, count);
          device.queue.submit([encoder.finish()]);
          await staging.mapAsync(GPUMapMode.READ, 0, count);
          observed.set(new Uint8Array(staging.getMappedRange(0, count)), position); staging.unmap();
        }
        actual = await digest(observed); location = entry.__view ? 'native-fused-gpu-view' : 'native-gpu-buffer';
      }
      checks.push({ name: record.name, location, expected: record.preparedSha256, actual, pass: actual === record.preparedSha256 });
      if (actual !== record.preparedSha256) throw new Error(`GPU materialization mismatch: ${record.name}`);
      console.log(`Materialization ${checks.length}/${weights.materializationBindings.length}: ${record.name}`);
    }
    return { schema: 'pooled.canonical-materialization.v1', checks, pass: true };
  } finally {
    output.destroy(); staging.destroy(); extent.destroy();
  }
}
