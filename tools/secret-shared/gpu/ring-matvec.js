/**
 * Exact Z_(2^64) ring matvec on WebGPU for Q4_K blocks. WGSL has no 64-bit integers, so every 64-bit value is two u32
 * limbs (lo, hi) with explicit carries; nothing is truncated or approximated.
 *
 * For a row and a 32-weight sub-block with scale A = round(d*sc*2^fw) and offset B = round(dmin*m*2^fw):
 *   contribution = sum_k 2^(16k) (A * S_k - B * T_k)   (mod 2^64)
 * where x = sum_k 2^(16k) x_k splits each input share into four 16-bit limbs, S_k = sum q * x_k (< 2^25) and
 * T_k = sum x_k (< 2^21) fit a u32 exactly, and A * S_k is a full 32x32->64 product built from 16-bit halves.
 * One workgroup per row; its threads split the row's blocks and reduce their 64-bit partial sums in shared memory.
 */
export const RING_MATVEC_WGSL = /* wgsl */ `
struct Params { rows: u32, cols: u32, n: u32, stride: u32, rowOffset: u32, fwScale: f32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> q4k: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<vec2<u32>>;
@group(0) @binding(3) var<uniform> p: Params;
const WG: u32 = 64u;
var<workgroup> partial: array<vec2<u32>, 64>;

fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let lo = a.x + b.x;
  return vec2<u32>(lo, a.y + b.y + select(0u, 1u, lo < a.x));
}
fn neg64(a: vec2<u32>) -> vec2<u32> { return add64(vec2<u32>(~a.x, ~a.y), vec2<u32>(1u, 0u)); }
fn mul32(a: u32, b: u32) -> vec2<u32> {
  let a0 = a & 0xffffu; let a1 = a >> 16u; let b0 = b & 0xffffu; let b1 = b >> 16u;
  let p00 = a0 * b0; let p01 = a0 * b1; let p10 = a1 * b0; let p11 = a1 * b1;
  let mid = p01 + p10;
  let midCarry = select(0u, 0x10000u, mid < p01);
  let lo = p00 + (mid << 16u);
  let loCarry = select(0u, 1u, lo < p00);
  return vec2<u32>(lo, p11 + (mid >> 16u) + midCarry + loCarry);
}
fn shl64(a: vec2<u32>, s: u32) -> vec2<u32> {
  if (s == 0u) { return a; }
  if (s >= 32u) { return vec2<u32>(0u, a.x << (s - 32u)); }
  return vec2<u32>(a.x << s, (a.y << s) | (a.x >> (32u - s)));
}
fn byteAt(base: u32, i: u32) -> u32 { return (q4k[base + i / 4u] >> ((i % 4u) * 8u)) & 0xffu; }
fn scaleMin(base: u32, j: u32) -> vec2<u32> {
  // ggml get_scale_min_k4 over the 12 scale bytes that start at byte 4 of the block.
  if (j < 4u) { return vec2<u32>(byteAt(base, 4u + j) & 63u, byteAt(base, 8u + j) & 63u); }
  let sc = (byteAt(base, 8u + j) & 0xfu) | ((byteAt(base, j) >> 6u) << 4u);
  let m = (byteAt(base, 8u + j) >> 4u) | ((byteAt(base, 4u + j) >> 6u) << 4u);
  return vec2<u32>(sc, m);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let row = wg.x + wg.y * 65535u;
  if (row >= p.rows) { return; }
  let blocks = p.cols / 256u;
  for (var c = 0u; c < p.n; c++) {
    var acc = vec2<u32>(0u, 0u);
    // Threads split the row's 32-weight sub-blocks (8 per Q4_K block), so even a 1024-wide row keeps 32 threads busy.
    for (var sb = tid; sb < blocks * 8u; sb += WG) {
      let b = sb / 8u; let j = sb % 8u;
      let base = (row * blocks + b) * 36u;
      let dm = unpack2x16float(q4k[base]);
      let sm = scaleMin(base, j);
      let A = u32(round(dm.x * f32(sm.x) * p.fwScale));
      let B = u32(round(dm.y * f32(sm.y) * p.fwScale));
      var S = vec4<u32>(0u); var T = vec4<u32>(0u);
      let qword = base + 4u + (j / 2u) * 8u;
      let shift = (j & 1u) * 4u;
      let xbase = c * p.cols + b * 256u + j * 32u;
      for (var w = 0u; w < 8u; w++) {
        let packed = q4k[qword + w];
        for (var e = 0u; e < 4u; e++) {
          let q = (packed >> (e * 8u + shift)) & 0xfu;
          let v = x[xbase + w * 4u + e];
          let limbs = vec4<u32>(v.x & 0xffffu, v.x >> 16u, v.y & 0xffffu, v.y >> 16u);
          S += limbs * q; T += limbs;
        }
      }
      for (var k = 0u; k < 4u; k++) {
        acc = add64(acc, shl64(add64(mul32(A, S[k]), neg64(mul32(B, T[k]))), 16u * k));
      }
    }
    partial[tid] = acc;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s = s / 2u) {
      if (tid < s) { partial[tid] = add64(partial[tid], partial[tid + s]); }
      workgroupBarrier();
    }
    if (tid == 0u) { y[c * p.stride + p.rowOffset + row] = partial[0]; }
    workgroupBarrier();
  }
}`;

const KEEP_ALIVE = [];

/** Opens a WebGPU device through the `webgpu` (Dawn) module, with the adapter's largest buffer limits. */
export async function openDevice() {
  const { create, globals } = await import(process.env.DOPPLER_NODE_WEBGPU_MODULE || 'webgpu');
  Object.assign(globalThis, globals);
  const gpu = create([]);
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw Error('No WebGPU adapter');
  const limits = adapter.limits;
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
    maxBufferSize: limits.maxBufferSize } });
  const info = adapter.info ?? {};
  // The Dawn instance and adapter must outlive the device: collecting them tears the device down under it.
  KEEP_ALIVE.push(gpu, adapter, device);
  return { device, adapterInfo: { vendor: info.vendor ?? null, architecture: info.architecture ?? null, device: info.device ?? null, description: info.description ?? null } };
}

/** A stage's Q4_K tensors resident on the GPU, and one ring matvec per projection group. */
export function createRingMatVec(device, { fw }) {
  const module = device.createShaderModule({ code: RING_MATVEC_WGSL });
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  const U = GPUBufferUsage;
  const upload = (bytes, usage) => {
    const buffer = device.createBuffer({ size: Math.ceil(bytes.byteLength / 4) * 4, usage: usage | U.COPY_DST });
    device.queue.writeBuffer(buffer, 0, bytes.buffer, bytes.byteOffset, bytes.byteLength);
    return buffer;
  };
  return {
    tensor(rows, cols, q4kBytes) { return { rows, cols, buffer: upload(q4kBytes, U.STORAGE) }; },
    /** x: BigUint64Array-compatible Uint32Array [n*cols*2] (lo, hi); returns Uint32Array [n*out*2]. */
    async run(tensors, x, n, out) {
      const cols = tensors[0].cols;
      const xBuffer = upload(x, U.STORAGE);
      const yBuffer = device.createBuffer({ size: n * out * 8, usage: U.STORAGE | U.COPY_SRC });
      const read = device.createBuffer({ size: n * out * 8, usage: U.MAP_READ | U.COPY_DST });
      const encoder = device.createCommandEncoder();
      const uniforms = [];
      let offset = 0;
      for (const t of tensors) {
        const params = new ArrayBuffer(32);
        new Uint32Array(params, 0, 5).set([t.rows, cols, n, out, offset]);
        new Float32Array(params, 20, 1)[0] = 2 ** fw;
        const uniform = device.createBuffer({ size: 32, usage: U.UNIFORM | U.COPY_DST });
        device.queue.writeBuffer(uniform, 0, params);
        uniforms.push(uniform);
        const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: t.buffer } }, { binding: 1, resource: { buffer: xBuffer } },
          { binding: 2, resource: { buffer: yBuffer } }, { binding: 3, resource: { buffer: uniform } }] });
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(Math.min(t.rows, 65535), Math.ceil(t.rows / 65535));
        pass.end();
        offset += t.rows;
      }
      encoder.copyBufferToBuffer(yBuffer, 0, read, 0, n * out * 8);
      device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const result = new Uint32Array(read.getMappedRange().slice(0));
      read.unmap();
      for (const buffer of [xBuffer, yBuffer, read, ...uniforms]) buffer.destroy();
      return result;
    }
  };
}
