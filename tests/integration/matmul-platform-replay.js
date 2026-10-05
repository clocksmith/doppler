/** Replay one captured Q4K projection with byte-identical operands on two GPUs. */
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { buildQ4KAccumulationDiagnostic } from '../kernels/q4k-accumulation-diagnostic.js';
import { dequantQ4_KRef } from '../kernels/reference/dequant.js';

const [capturePath, installedPackage, reploidRoot, output, phase = 'prefill'] = process.argv.slice(2);
assert(capturePath && installedPackage && reploidRoot && output && process.env.REPLOID_EXECUTOR_WS);
assert(['prefill', 'decode'].includes(phase));
const captureBytes = await readFile(capturePath), capture = JSON.parse(captureBytes);
const projection = capture.captures.flatMap(c => c.records)
  .find(r => r.boundary === 'projection' && (phase === 'decode' ? r.M === 1 : r.M > 1));
assert(projection, `Missing ${phase} projection`);
const geometry = projection.config.variantMetadata;
assert(['row-column-tile', 'column-block'].includes(geometry.dispatchGeometry), 'Unsupported captured dispatch geometry');
const originalShader = await readFile(resolve(installedPackage, 'src/gpu/kernels', projection.config.shaderFile), 'utf8');
const candidate = process.env.DOPPLER_Q4K_DIAGNOSTIC ?? null;
let shader = candidate ? buildQ4KAccumulationDiagnostic(originalShader, candidate) : originalShader;
const observeLanes = process.env.DOPPLER_Q4K_OBSERVE_LANES === '1';
if (observeLanes) {
  assert(candidate === 'fma-four-lanes');
  shader = shader.replace('let accum = results[m];',
    'let accum = results[m];\nobserved_lanes[row * u.N + col] = accum;')
    + '\n@group(0) @binding(4) var<storage, read_write> observed_lanes: array<vec4<f32>>;';
}
const { chromium } = createRequire(resolve(reploidRoot, 'package.json'))('playwright');
const hash = value => createHash('sha256').update(value).digest('hex');
const receipt = { scope: 'Identical captured operands; isolated operation, not model acceptance',
  candidate, observeLanes, originalShaderSha256: hash(originalShader), sourceSubstitution: candidate !== null,
  captureSha256: hash(captureBytes), archiveSha256: capture.archiveSha256, shaderSha256: hash(shader),
  contract: { M: projection.M, N: projection.N, K: projection.K, variant: projection.variant,
    entryPoint: projection.config.entryPoint, constants: projection.constants }, results: [] };
for (const platform of ['mac', 'linux']) {
  const browser = platform === 'mac'
    ? await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=metal'] })
    : await chromium.connect(process.env.REPLOID_EXECUTOR_WS);
  const context = await browser.newContext();
  try {
    const page = await context.newPage();
    await page.goto('http://localhost:8000/config/chat-files.json');
    const result = await page.evaluate(async ({ projection, shader, observeLanes }) => {
      const adapter = await navigator.gpu.requestAdapter();
      const device = await adapter.requestDevice({ requiredFeatures: projection.config.requires });
      const owned = [];
      const buffer = (size, usage) => { const b = device.createBuffer({ size, usage }); owned.push(b); return b; };
      try {
        const module = device.createShaderModule({ code: shader });
        const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error');
        if (errors.length) throw Error(errors.map(m => m.message).join('\n'));
        const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: {
          module, entryPoint: projection.config.entryPoint, constants: projection.constants ?? {} } });
        const u = new ArrayBuffer(32), view = new DataView(u);
        view.setUint32(0, projection.M, true); view.setUint32(4, projection.N, true);
        view.setUint32(8, projection.K, true); view.setFloat32(12, projection.alpha, true);
        view.setUint32(16, projection.K / 256, true);
        const uniform = buffer(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        device.queue.writeBuffer(uniform, 0, u);
        const tensors = projection.tensors.map(t => {
          const b = buffer(t.bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
          if (t.role !== 'output') device.queue.writeBuffer(b, 0, Uint8Array.from(atob(t.data), c => c.charCodeAt(0)));
          return b;
        });
        const outBytes = projection.M * projection.N * 4;
        const lanes = observeLanes ? buffer(outBytes * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC) : null;
        const laneReadback = observeLanes ? buffer(outBytes * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ) : null;
        const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
          entries: [uniform, ...tensors, ...(lanes ? [lanes] : [])]
            .map((buffer, binding) => ({ binding, resource: { buffer } })) });
        const staging = buffer(outBytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, group);
        const geometry = projection.config.variantMetadata;
        const cols = projection.constants?.COLS_PER_WG ?? geometry.colsPerWg;
        if (!Number.isInteger(cols) || cols < 1) throw Error('Missing captured column geometry');
        pass.dispatchWorkgroups(Math.ceil(projection.N / cols),
          geometry.dispatchGeometry === 'row-column-tile' ? Math.ceil(projection.M / geometry.tileM) : 1);
        pass.end();
        encoder.copyBufferToBuffer(tensors[2], 0, staging, 0, outBytes);
        if (lanes) encoder.copyBufferToBuffer(lanes, 0, laneReadback, 0, outBytes * 4);
        device.queue.submit([encoder.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const bytes = new Uint8Array(staging.getMappedRange().slice(0)); staging.unmap();
        let text = '';
        for (let i = 0; i < bytes.length; i += 16384) text += String.fromCharCode(...bytes.subarray(i, i + 16384));
        let laneData = null;
        if (laneReadback) {
          await laneReadback.mapAsync(GPUMapMode.READ);
          const bytes = new Uint8Array(laneReadback.getMappedRange().slice(0)); laneReadback.unmap();
          let encoded = '';
          for (let i = 0; i < bytes.length; i += 16384) encoded += String.fromCharCode(...bytes.subarray(i, i + 16384));
          laneData = btoa(encoded);
        }
        return { data: btoa(text), laneData, vendor: adapter.info.vendor, architecture: adapter.info.architecture };
      } finally { for (const b of owned) b.destroy(); device.destroy(); }
    }, { projection, shader, observeLanes });
    receipt.results.push({ platform, browser: browser.version(), ...result });
    console.log(JSON.stringify({ platform, completed: true }));
  } finally { await context.close(); await browser.close(); }
}
const values = text => { const b = Buffer.from(text, 'base64'); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };
const compare = (a, b) => {
  assert.equal(a.length, b.length); let maxDifference = 0, differentElements = 0;
  for (let i = 0; i < a.length; i++) {
    assert(Number.isFinite(a[i]) && Number.isFinite(b[i]), 'Every observed value must be finite');
    maxDifference = Math.max(maxDifference, Math.abs(a[i] - b[i])); if (a[i] !== b[i]) differentElements++;
  }
  return { elements: a.length, differentElements, maxDifference };
};
receipt.crossPlatform = compare(...receipt.results.map(r => values(r.data)));
if (observeLanes) receipt.crossPlatformLanes = compare(...receipt.results.map(r => values(r.laneData)));
receipt.macCaptureAgreement = compare(values(receipt.results[0].data), values(projection.tensors.find(t => t.role === 'output').data));
// Independent host reference is diagnostic only; these values never feed inference.
const input = values(projection.tensors.find(t => t.role === 'input').data);
const packed = Uint8Array.from(Buffer.from(projection.tensors.find(t => t.role === 'weight').data, 'base64'));
assert.equal(projection.K % 256, 0, 'Reference currently requires complete Q4K blocks');
const weights = dequantQ4_KRef(packed, projection.N * projection.K / 256);
const reference = new Float64Array(projection.M * projection.N);
for (let m = 0; m < projection.M; m++) {
  for (let n = 0; n < projection.N; n++) {
    let sum = 0;
    for (let k = 0; k < projection.K; k++) sum += input[m * projection.K + k] * weights[n * projection.K + k];
    reference[m * projection.N + n] = sum * projection.alpha;
  }
}
receipt.reference = { scope: 'Independent Q4K unpack to F32; Float64 dot-product accumulation',
  sha256: hash(Buffer.from(reference.buffer)), comparisons: receipt.results.map(result => {
    const actual = values(result.data); let squaredError = 0;
    for (let i = 0; i < actual.length; i++) squaredError += (actual[i] - reference[i]) ** 2;
    return { platform: result.platform, ...compare(actual, reference), rmsError: Math.sqrt(squaredError / actual.length) };
  }) };
await writeFile(output, JSON.stringify(receipt));
console.log(JSON.stringify({ crossPlatform: receipt.crossPlatform, crossPlatformLanes: receipt.crossPlatformLanes,
  macCaptureAgreement: receipt.macCaptureAgreement,
  reference: receipt.reference }));
