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
const source = await readFile(new URL('../../src/gpu/kernels/codegen/includes/paired-half-store.wgsl.inc', import.meta.url), 'utf8');
const inputs=[], references=[];
const view=new DataView(new ArrayBuffer(4));
const bits=value=>{view.setFloat32(0,value,true);return view.getUint32(0,true)};
const float=value=>{view.setUint32(0,value,true);return view.getFloat32(0,true)};
function halfValue(encoded) {
  const exponent = encoded >>> 10;
  const significand = encoded & 1023;
  return exponent === 0 ? significand * 2 ** -24
    : (1024 + significand) * 2 ** (exponent - 25);
}
const positiveHalf = Array.from({ length: 0x7c01 }, (_, index) => halfValue(index));
// Exact signed integer multiples of 2^-149: every finite F32 is represented
// without Number addition, including a residual too small to change an F64 sum.
function f32Units(value) {
  const raw = bits(value), exponent = (raw >>> 23) & 255;
  assert(exponent < 255, 'Finite paired operands required.');
  const significand = BigInt((raw & 0x7fffff) | (exponent ? 0x800000 : 0));
  const magnitude = exponent ? significand << BigInt(exponent - 1) : significand;
  return raw >>> 31 ? -magnitude : magnitude;
}
const halfUnits = positiveHalf.map(f32Units);
function nearestHalf(high, low) {
  const exact = f32Units(high) + f32Units(low);
  // A zero pair preserves its high-component sign, including expansion(-0,+0).
  // Nonzero cancellation is outside the normalized-pair domain.
  const negative = exact < 0n || (high === 0 && low === 0 && Object.is(high, -0));
  const magnitude = exact < 0n ? -exact : exact;
  let lo = 0, hi = 0x7c00;
  while (lo + 1 < hi) {
    const middle = (lo + hi) >>> 1;
    if (halfUnits[middle] <= magnitude) lo = middle;
    else hi = middle;
  }
  const lowerDistance = magnitude - halfUnits[lo];
  const upperDistance = halfUnits[hi] - magnitude;
  const selected = lowerDistance < upperDistance ? lo
    : upperDistance < lowerDistance ? hi : (lo & 1) === 0 ? lo : hi;
  return selected | (negative ? 0x8000 : 0);
}

for(let lower=0;lower<0x7c00;lower++) {
 const midpoint=(positiveHalf[lower]+positiveHalf[lower+1])/2, encoded=bits(midpoint);
 for(const sign of [-1,1]) for(const adjacent of [-1,0,1]) for(const residualSign of [-1,0,1]) {
  const high=sign*float(encoded+adjacent),ulp=2**(((bits(high)>>>23)&255)-150),low=residualSign*ulp/4;
  inputs.push(high,low,0,0); references.push(nearestHalf(high,low));
 }
}
const quarterUlpCount = inputs.length / 4;
for (let lower = 0; lower < 0x7c00; lower++) {
  const midpoint = (positiveHalf[lower] + positiveHalf[lower + 1]) / 2;
  for (const sign of [-1, 1]) for (const residualSign of [-1, 1]) {
    const high = sign * midpoint, low = residualSign * float(1);
    inputs.push(high, low, 0, 0); references.push(nearestHalf(high, low));
  }
}
for (const high of [0, -0]) for (const low of [0, -0]) {
  inputs.push(high, low, 0, 0); references.push(nearestHalf(high, low));
}
for (const high of [float(1), -float(1)]) {
  inputs.push(high, 0, 0, 0); references.push(nearestHalf(high, 0));
}
const count = inputs.length / 4;
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const receipt = { schema: 'doppler-direct-paired-half-physical-v1', shaderSha256: hash(source),
  inputSha256: hash(Buffer.from(new Float32Array(inputs).buffer)), count, quarterUlpCount, tinyResidualMidpoints: 0x7c00 * 4,
  acceptance: { exactBinary16Bits: true, zeroPairSign: 'preserve high component' },
  scope: 'Every finite half midpoint through overflow, both signs, nearest three F32 highs and quarter-ULP residual signs; exact BigInt dyadic half-distance/tie reference including minimum-F32-subnormal residual at every midpoint.' };
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
 let v=input[id.x];output[id.x*2u]=vec4<f32>(f32(pair_half_bits(v.xy)),0.0,0.0,0.0);
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
  for(let i=0;i<count;i++) {
    const actual=result.values[i*8];
    if(actual!==references[i])receipt.failures.push({i,input:inputs.slice(i*4,i*4+2),expected:references[i],actual});
  }
  receipt.passed=receipt.failures.length===0&&result.errors.length===0;
} catch(error) { receipt.error=error.stack; throw error; }
finally { await browser?.close();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));await writeFile(destination,JSON.stringify(receipt,null,2)+'\n'); }
console.log(JSON.stringify({...receipt,failures:receipt.failures.length}));
assert(receipt.passed,'Direct paired F16 rounding controls failed.');
