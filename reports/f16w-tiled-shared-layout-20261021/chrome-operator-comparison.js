import fs from 'node:fs/promises';
import { chromium } from 'playwright';
import http from 'node:http';
const server = http.createServer((q, r) => { r.setHeader('content-type', 'text/html'); r.end('<!doctype html><title>k</title>'); });
await new Promise((res) => server.listen(0, '127.0.0.1', res));
// usage: node chrome-operator-comparison.js <previous.wgsl> <candidate.wgsl> <out.json>
const [previousPath, candidatePath, outputPath] = process.argv.slice(2);
const [oldSrc, newSrc] = await Promise.all([fs.readFile(previousPath, 'utf8'), fs.readFile(candidatePath, 'utf8')]);
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu', '--enable-webgpu-developer-features', '--ignore-gpu-blocklist', '--use-angle=vulkan', '--enable-features=Vulkan', '--disable-vulkan-surface'] });
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${server.address().port}/`);
const out = await page.evaluate(async ({ oldSrc, newSrc }) => {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }) ?? await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('no adapter; isSecureContext=' + isSecureContext);
  const device = await adapter.requestDevice({ requiredFeatures: ['shader-f16', 'timestamp-query'].filter((f) => adapter.features.has(f)) });
  const mk = (code) => device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
  const pipes = { old: mk(oldSrc), new: mk(newSrc) };
  let seed = 12345;
  const rnd = () => { seed = (seed * 1103515245 + 12345) >>> 0; return (seed / 4294967296) * 2 - 1; };
  const f32ToF16 = (v) => { const f = new Float32Array([v]); const x = new Uint32Array(f.buffer)[0]; const s = (x >>> 16) & 0x8000; let e = ((x >>> 23) & 0xff) - 127 + 15; let m = (x >>> 13) & 0x3ff; if (e <= 0) return s; if (e >= 31) return s | 0x7c00; return s | (e << 10) | m; };
  async function run(name, M, N, K, A, B, reps) {
    const buf = (data, usage) => { const b = device.createBuffer({ size: Math.max(16, Math.ceil(data.byteLength / 4) * 4), usage, mappedAtCreation: true }); new Uint8Array(b.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength)); b.unmap(); return b; };
    const u = new ArrayBuffer(32); new Uint32Array(u, 0, 3).set([M, N, K]); new Float32Array(u, 12, 1)[0] = 1; new Uint32Array(u, 16, 1)[0] = 1;
    const ub = buf(new Uint8Array(u), GPUBufferUsage.UNIFORM);
    const ab = buf(A, GPUBufferUsage.STORAGE); const bb = buf(B, GPUBufferUsage.STORAGE);
    const cb = device.createBuffer({ size: M * N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const rb = device.createBuffer({ size: M * N * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const pipe = pipes[name];
    const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [ub, ab, bb, cb].map((b, i) => ({ binding: i, resource: { buffer: b } })) });
    const qs = device.createQuerySet({ type: 'timestamp', count: 2 * reps });
    const qb = device.createBuffer({ size: 16 * reps, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const qr = device.createBuffer({ size: 16 * reps, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    for (let r = 0; r < reps; r += 1) {
      const pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * r, endOfPassWriteIndex: 2 * r + 1 } });
      pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(Math.ceil(M / 64), Math.ceil(N / 64)); pass.end();
    }
    enc.resolveQuerySet(qs, 0, 2 * reps, qb, 0); enc.copyBufferToBuffer(qb, 0, qr, 0, 16 * reps); enc.copyBufferToBuffer(cb, 0, rb, 0, M * N * 4);
    device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ); await qr.mapAsync(GPUMapMode.READ);
    const bytes = new Uint8Array(rb.getMappedRange()).slice();
    const t = new BigUint64Array(qr.getMappedRange()); const ms = [];
    for (let r = 0; r < reps; r += 1) ms.push(Number(t[2 * r + 1] - t[2 * r]) / 1e6);
    for (const b of [ub, ab, bb, cb, rb, qb, qr]) b.destroy(); qs.destroy();
    ms.sort((a, b) => a - b);
    return { bytes, medianMs: ms[Math.floor(ms.length / 2)] };
  }
  const rows = [];
  const shapes = [[1, 1, 1], [3, 5, 7], [64, 64, 16], [65, 63, 17], [102, 3072, 1024], [102, 1024, 3072], [613, 3072, 1024], [613, 1024, 3072], [613, 2048, 1024], [130, 151, 2000], [513, 4096, 2560]];
  for (const [M, N, K] of shapes) {
    const A = new Float32Array(M * K).map(() => rnd() * 4);
    const B = new Uint16Array(N * K).map(() => f32ToF16(rnd() * 0.25));
    const reps = M * N * K > 1e8 ? 20 : 3;
    const o = await run('old', M, N, K, A, B, reps);
    const n = await run('new', M, N, K, A, B, reps);
    let diff = 0; for (let i = 0; i < o.bytes.length; i += 1) if (o.bytes[i] !== n.bytes[i]) diff += 1;
    rows.push({ M, N, K, identicalBytes: diff === 0, differingBytes: diff, oldMs: o.medianMs, newMs: n.medianMs, gflopsNew: (2 * M * N * K) / (n.medianMs * 1e6) });
  }
  return { adapter: adapter.info ? { vendor: adapter.info.vendor, architecture: adapter.info.architecture, description: adapter.info.description } : null, rows };
}, { oldSrc, newSrc });
await browser.close();
server.close();
console.log(JSON.stringify(out.adapter));
for (const r of out.rows) console.log(`${r.M}x${r.N}x${r.K} identical=${r.identicalBytes} old=${r.oldMs.toFixed(3)}ms new=${r.newMs.toFixed(3)}ms speedup=${(r.oldMs / r.newMs).toFixed(2)}x newGFLOPS=${r.gflopsNew.toFixed(0)}`);
await fs.writeFile(outputPath, JSON.stringify(out, null, 2));
