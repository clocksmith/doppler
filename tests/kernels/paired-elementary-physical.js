// Physical operator qualification; no model or captured-output reference.
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname } from 'node:path';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { chromium } from 'playwright';

const destination = process.argv[2];
if (!destination) throw new Error('Provide a receipt destination.');
await mkdir(dirname(destination), { recursive: true });
const source = (await Promise.all(['compensated-f32', 'paired-elementary', 'paired-half-store'].map(name => readFile(new URL(`../../src/gpu/kernels/codegen/includes/${name}.wgsl.inc`, import.meta.url), 'utf8')))).join('\n');
const inputs = [];
let seed = 624901;
function random() { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return (seed >>> 0) / 2 ** 32; }
const f = Math.fround;
for (let i=0;i<2048;i++) {
 const a=f((random()-0.5)*80), b=f((random()-0.5)*16000);
 inputs.push(a,f(a*(random()-0.5)*2**-24),b,f(b*(random()-0.5)*2**-24));
}
for (const a of [-40,-20,-1,-(2**-20),0,2**-20,1,20,40]) inputs.push(a,0,8192,0);
const count = inputs.length / 4;
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const receipt = { schema: 'doppler-paired-elementary-physical-v1', shaderSha256: hash(source),
  inputSha256: hash(Buffer.from(new Float32Array(inputs).buffer)), count, seed: 624901,
  acceptance: { normalizedAbsoluteError: 2e-12 },
  scope: 'Independent Number/F64 elementary references over frozen inputs; not model qualification. Direct half exhaustive control is separate.' };
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
 let v = input[id.x]; let a=v.xy; let b=v.zw;
 let positive=pair_add(pair_mul(a,a),vec2<f32>(1.0,0.0));
 output[id.x*6u] = vec4<f32>(pair_exp(a),pair_log(positive));
 output[id.x*6u+1u] = vec4<f32>(pair_sqrt(positive),pair_div(a,positive));
 output[id.x*6u+2u] = vec4<f32>(pair_sigmoid(a),pair_silu(a));
 output[id.x*6u+3u] = vec4<f32>(pair_softplus(a),pair_mul(a,b));
 output[id.x*6u+4u] = pair_sin_cos(b);
 output[id.x*6u+5u] = vec4<f32>(f32(pair_half_bits(a)),0.0,0.0,0.0);
}` });
      const bad = (await module.getCompilationInfo()).messages.filter(row => row.type === 'error');
      if (bad.length) throw new Error(bad.map(row => row.message).join('\n'));
      const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
      const input = alloc(inputs.length*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
      const output = alloc(count*96,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
      staging = alloc(count*96,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
      device.queue.writeBuffer(input,0,new Float32Array(inputs));
      const group = device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[input,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
      const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();
      pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(count/64));pass.end();
      encoder.copyBufferToBuffer(output,0,staging,0,count*96);device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      return {values:Array.from(new Float32Array(staging.getMappedRange().slice(0))),errors,adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture}};
    } finally {
      if(staging?.mapState==='mapped')staging.unmap();
      await device.queue.onSubmittedWorkDone();for(const buffer of owned)buffer.destroy();device.destroy();
    }
  }, {source,inputs,count});
  receipt.adapter=result.adapter;receipt.browser=browser.version();receipt.errors=result.errors;receipt.failures=[];
  receipt.maxError=0;receipt.maxByOperation={};
  const names=['exp','log','sqrt','div','sigmoid','silu','softplus','multiply','sin','cos'];
  for(let i=0;i<count;i++) {
    const [ah,al,bh,bl]=inputs.slice(i*4,i*4+4),a=ah+al,b=bh+bl,p=a*a+1;
    const expected=[Math.exp(a),Math.log(p),Math.sqrt(p),a/p,1/(1+Math.exp(-a)),a/(1+Math.exp(-a)),Math.max(a,0)+Math.log1p(Math.exp(-Math.abs(a))),a*b,Math.sin(b),Math.cos(b)];
    for(let lane=0;lane<expected.length;lane++) {
      const high=result.values[i*24+lane*2],low=result.values[i*24+lane*2+1];
      const error=Math.abs(high+low-expected[lane])/Math.max(1,Math.abs(expected[lane]));
      receipt.maxError=Math.max(receipt.maxError,error);
      receipt.maxByOperation[names[lane]]=Math.max(receipt.maxByOperation[names[lane]]||0,error);
      if(!Number.isFinite(error)||error>receipt.acceptance.normalizedAbsoluteError) receipt.failures.push({i,operation:names[lane],input:[ah,al,bh,bl],output:[high,low],expected:expected[lane],error});
    }
  }
  receipt.passed=receipt.failures.length===0&&result.errors.length===0;
} catch(error) { receipt.error=error.stack; throw error; }
finally { await browser?.close();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));await writeFile(destination,JSON.stringify(receipt,null,2)+'\n'); }
console.log(JSON.stringify({...receipt,failures:receipt.failures.length}));
assert(receipt.passed,'Paired elementary operator controls failed.');
