// Physical operator qualification; no model or captured-output reference.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { chromium } from 'playwright';

const destination = process.argv[2];
if (!destination) throw new Error('Provide a receipt destination.');
const source = await readFile(new URL('../../src/gpu/kernels/codegen/includes/compensated-f32.wgsl.inc', import.meta.url), 'utf8');
const inputs = [];
let seed = 624901;
function random() { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return (seed >>> 0) / 2 ** 32; }
const f = Math.fround;
for (let i = 0; i < 20000; i++) {
  const a = f((random() - 0.5) * 512);
  const b = f((random() - 0.5) * 512);
  inputs.push(a, f(a * (random() - 0.5) * 2 ** -24), b, f(b * (random() - 0.5) * 2 ** -24));
}
for (const sign of [-1, 1]) for (const a of [0, 2 ** -126, 2 ** -20, 1, 128]) {
  inputs.push(sign * a, f(sign * a * 2 ** -25), -sign * a, f(sign * a * 2 ** -25));
}
for (const sign of [-1, 1]) {
  inputs.push(0, sign * 2 ** -149, 0, 0);
  inputs.push(sign * 2 ** -126, 0, -sign * (2 ** -126 - 2 ** -149), 0);
}
const count = inputs.length / 4;
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const receipt = { schema: 'doppler-compensated-state-physical-v1', shaderSha256: hash(source),
  inputSha256: hash(Buffer.from(new Float32Array(inputs).buffer)), count, seed: 624901,
  acceptance: { productAndSumResidual: 'exact binary32 bits (signed zero equivalent)', pairNormalizedAbsoluteError: 2e-13 },
  scope: 'Finite normal recurrent-scale pair operators; cancellation and min-normal edge controls. Not whole-model acceptance.' };
const server = createServer((_req, res) => res.end('<!doctype html>'));
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu',
    ...(process.platform === 'darwin' ? ['--use-angle=metal'] : ['--enable-features=Vulkan', '--use-angle=vulkan', '--disable-gpu-sandbox'])] });
  const page = await browser.newPage(); await page.goto(`http://127.0.0.1:${server.address().port}`);
  const result = await page.evaluate(async ({ source, inputs, count }) => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter || adapter.info.isFallbackAdapter) throw new Error('Physical adapter required.');
    const device = await adapter.requestDevice(), owned = [], errors = [];
    device.addEventListener('uncapturederror', event => errors.push(event.error.message));
    const alloc = (size, usage) => { const b = device.createBuffer({ size, usage }); owned.push(b); return b; };
    let staging;
    try {
      const module = device.createShaderModule({ code: source + `
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
 if (id.x >= arrayLength(&input)) { return; }
 let v = input[id.x]; let a = v.x; let b = v.z;
 let sum = a + b; let product = a * b;
 output[id.x*2u] = vec4<f32>(sum, addition_residual(a,b,sum),product,product_residual(a,b,product));
 output[id.x*2u+1u] = vec4<f32>(pair_add(v.xy,v.zw),pair_mul_scalar(v.xy,b));
}` });
      const bad = (await module.getCompilationInfo()).messages.filter(row => row.type === 'error');
      if (bad.length) throw new Error(bad.map(row => row.message).join('\n'));
      const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
      const input = alloc(inputs.length*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
      const output = alloc(count*32,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
      staging = alloc(count*32,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
      device.queue.writeBuffer(input,0,new Float32Array(inputs));
      const group = device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[input,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
      const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();
      pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(count/64));pass.end();
      encoder.copyBufferToBuffer(output,0,staging,0,count*32);device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      return {values:Array.from(new Float32Array(staging.getMappedRange().slice(0))),errors,adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture}};
    } finally {
      if(staging?.mapState==='mapped')staging.unmap();
      await device.queue.onSubmittedWorkDone();for(const buffer of owned)buffer.destroy();device.destroy();
    }
  }, {source,inputs,count});
  receipt.adapter=result.adapter;receipt.browser=browser.version();receipt.errors=result.errors;receipt.failures=[];
  receipt.maxPairError=0;
  for(let i=0;i<count;i++) {
    const [a,al,b,bl]=inputs.slice(i*4,i*4+4),o=result.values.slice(i*8,i*8+8);
    const expected=[f((a+b)-o[0]),f(a*b-o[2])];
    const references=[(a+al)+(b+bl),(a+al)*b];
    const errors=[Math.abs((o[4]+o[5])-references[0])/Math.max(1,Math.abs(references[0])),Math.abs((o[6]+o[7])-references[1])/Math.max(1,Math.abs(references[1]))];
    receipt.maxPairError=Math.max(receipt.maxPairError,...errors);
    if(o[1]!==expected[0]||o[3]!==expected[1]||errors.some(error=>error>2e-13)
      || references.some((value,lane)=>Math.abs(value)<2**-126 && f(value)!==o[4+lane*2]+o[5+lane*2]))receipt.failures.push({i,inputs:[a,al,b,bl],output:o,expected,errors});
  }
  receipt.passed=receipt.failures.length===0&&result.errors.length===0;
} catch(error) { receipt.error=error.stack; throw error; }
finally { await browser?.close();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));await writeFile(destination,JSON.stringify(receipt,null,2)+'\n'); }
console.log(JSON.stringify({...receipt,failures:receipt.failures.length}));
assert(receipt.passed,'Compensated state operator controls failed.');
