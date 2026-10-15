import assert from 'node:assert/strict';
import { writeFile } from 'node:fs/promises';
import { create, globals } from 'webgpu';

Object.assign(globalThis, globals);
const gpu = create(['enable-dawn-features=allow_unsafe_apis']);
const adapter = await gpu.requestAdapter();
assert(adapter, 'Physical WebGPU adapter required');
const device = await adapter.requestDevice();
const buffers = [];
try {
  const code = `
@group(0) @binding(0) var<storage, read> packed: array<u32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
fn decode_e4m3fn(byte: u32) -> f32 {
  let sign = select(1.0, -1.0, (byte & 128u) != 0u);
  let exponent = (byte >> 3u) & 15u;
  let mantissa = byte & 7u;
  if (exponent == 15u && mantissa == 7u) { return 0.0; }
  if (exponent == 0u) { return sign * f32(mantissa) * 0.001953125; }
  return sign * bitcast<f32>(((exponent + 120u) << 23u) | (mantissa << 20u));
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < 256u) {
    let byte = (packed[i / 4u] >> ((i % 4u) * 8u)) & 255u;
    result[i] = decode_e4m3fn(byte);
    if ((byte & 127u) == 127u) { result[258u + (byte >> 7u)] = 1.0; }
  }
  if (i == 0u) {
    result[256] = f32(dot4I8Packed(0x04fd0201u, 0x08070605u));
    let word = 0x30c44038u;
    var sum = 0.0;
    for (var j = 0u; j < 4u; j++) {
      sum += decode_e4m3fn((word >> (j * 8u)) & 255u) * f32(j + 2u);
    }
    result[257] = sum;
  }
}`;
  const module = device.createShaderModule({ code });
  const info = await module.getCompilationInfo();
  assert.equal(info.messages.filter(x => x.type === 'error').length, 0, JSON.stringify(info.messages));
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  function buffer(size, usage) {
    const b = device.createBuffer({ size, usage });
    buffers.push(b);
    return b;
  }
  const input = buffer(256, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const output = buffer(260 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const readback = buffer(260 * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(input, 0, Uint8Array.from({ length: 256 }, (_, i) => i));
  const bindings = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: input } }, { binding: 1, resource: { buffer: output } },
  ] });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0, bindings); pass.dispatchWorkgroups(4); pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, 260 * 4);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await readback.mapAsync(GPUMapMode.READ);
  const values = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  for (let i = 0; i < 256; i++) {
    const exponent = (i >> 3) & 15;
    const mantissa = i & 7;
    if (exponent === 15 && mantissa === 7) { assert.equal(values[258 + (i >> 7)], 1); continue; }
    const magnitude = exponent === 0 ? mantissa * 2 ** -9 : (1 + mantissa / 8) * 2 ** (exponent - 7);
    const expected = i & 128 ? -magnitude : magnitude;
    assert(values[i] === expected, `FP8 decoding mismatch for byte ${i}`);
  }
  assert.equal(values[256], 28);
  assert.equal(values[257], -1.5);
  const receipt = { passed: true, adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture },
    features: [...adapter.features], wgslFeatures: [...gpu.wgslLanguageFeatures],
    fp8Encoding: 'E4M3FN', finitePatternsDecoded: 254, nonfinitePatternsFlagged: 2, packedIntegerDot: values[256], packedFp8WiderDot: values[257],
    scope: 'Tiny physical shader probe: u32-packed bytes, FP8 decoded to f32, and signed packed integer dot product. No native FP8, whole-model FP8, performance or model-quality claim.' };
  if (process.argv[2]) await writeFile(process.argv[2], JSON.stringify(receipt, null, 2));
  console.log(JSON.stringify(receipt, null, 2));
} finally {
  await device.queue.onSubmittedWorkDone().catch(() => {});
  for (const buffer of buffers) buffer.destroy();
  device.destroy();
}
