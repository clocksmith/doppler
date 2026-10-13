// Isolated candidate attention dispatch against the existing independent F64 cases.
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
const [root,destination]=process.argv.slice(2), here=import.meta.dirname;
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const proposal=JSON.parse(readFileSync(resolve(here,'proposal.json')));
const source=readFileSync(resolve(here,'candidate.wgsl'),'utf8');assert.equal(hash(source),proposal.candidate);
assert.equal(hash(readFileSync(resolve(root,'node_modules/webgpu/dist/linux-x64.dawn.node'))),'fbfe599055b469963e4c9bfc900a702843db962dcaba0799b3c81ae33fa43db1');
const cg=readFileSync('/proc/self/cgroup','utf8').trim().split(':').at(-1);assert(cg.endsWith('/reploid-rms-repair-20261013.scope'));
writeFileSync('/sys/fs/cgroup'+cg+'/memory.oom.group','1');
for(const [name,value] of Object.entries({'memory.high':'6442450944','memory.max':'8589934592','memory.swap.max':'0','memory.oom.group':'1'}))assert.equal(readFileSync('/sys/fs/cgroup'+cg+'/'+name,'utf8').trim(),value);
const receipt={schema:'doppler.rms-single-div-probe/v1',proposal,pid:process.pid,events:[],rows:[],tolerance:2e-12,peakGpuBufferBytes:0};
const persist=()=>writeFileSync(destination,JSON.stringify(receipt,null,2)+'\n');
const mark=(phase,extra={})=>{receipt.events.push({phase,at:new Date().toISOString(),...extra});persist();};
const bounded=async(promise,ms,label)=>{let timer;try{return await Promise.race([promise,new Promise((_,reject)=>{timer=setTimeout(()=>reject(Error(label+' deadline; ownership unknown')),ms);})]);}finally{clearTimeout(timer);}};
const {createNativeGpuHost}=await import(pathToFileURL(resolve(root,'server/native-participant.js')));
const host=createNativeGpuHost(()=>import(pathToFileURL(resolve(root,'node_modules/webgpu/index.js'))));
const cases = [];
let seed = 624901;
function random() { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return (seed >>> 0) / 2 ** 32; }
for (const hidden of [128, 256, 1024]) for (const pre of [false, true]) {
  const input = [], residual = [], weight = [];
  for (let i = 0; i < hidden * 2; i++) {
    const hi = Math.fround((random() - .5) * 4), low = Math.fround((random() - .5) * 2 ** -25);
    input.push(hi, low); residual.push(Math.fround((random() - .5) * .2), Math.fround((random() - .5) * 2 ** -28));
  }
  for (let i = 0; i < hidden; i++) weight.push(Math.fround(.5 + random()));
  cases.push({ hidden, tokens: 2, input, residual, weight, eps: pre ? 1e-6 : 1e-5, pre, post: !pre, offset: pre });
}

receipt.fixtureSha256=hash(JSON.stringify(cases));receipt.cases=[];let operation,aborted=false;const guard=()=>{if(aborted)throw Error('Stopped after deadline');};
try { operation=(async()=>{await host.open();guard();mark('host-open');
  const physical = await (async ({ source, cases }) => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter || adapter.info.isFallbackAdapter) throw new Error('Physical adapter required.');
    const device = await adapter.requestDevice(), errors = [], results = [];
    device.addEventListener('uncapturederror', event => errors.push(event.error.message));
    device.lost.then(info=>mark('device-lost',{reason:info.reason,message:info.message}));
    const create=device.createComputePipelineAsync.bind(device);
    device.createComputePipelineAsync=async descriptor=>{guard();mark('pipeline-start',{constants:descriptor.compute.constants});const result=await create(descriptor);guard();mark('pipeline-complete',{constants:descriptor.compute.constants});return result;};
    const allocate=device.createBuffer.bind(device);let bytes=0;
    device.createBuffer=descriptor=>{bytes+=descriptor.size;assert(bytes<=proposal.admission.maxGpuBufferBytes);receipt.peakGpuBufferBytes=Math.max(receipt.peakGpuBufferBytes,bytes);return allocate(descriptor);};
    try {
      const module = device.createShaderModule({ code: source });
      const bad = (await module.getCompilationInfo()).messages.filter(row => row.type === 'error');
      if (bad.length) throw new Error(bad.map(row => row.message).join('\n'));
      for (const fixture of cases) {
        const owned = [], mapped = [];
        const alloc = (size, usage) => { const buffer = device.createBuffer({ size, usage }); owned.push(buffer); return buffer; };
        try {
          const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main', constants: {
            WORKGROUP_SIZE: 64, WEIGHT_F16: false, WEIGHT_OFFSET: fixture.offset,
            PRE_RESIDUAL: fixture.pre, POST_RESIDUAL: fixture.post, OUTPUT_PRENORM: fixture.pre,
          } } });
          const uniform = alloc(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
          const uniformBytes = new ArrayBuffer(32), view = new DataView(uniformBytes);
          view.setUint32(0, fixture.hidden, true); view.setUint32(4, fixture.tokens, true);
          view.setFloat32(8, Math.fround(fixture.eps), true); view.setFloat32(12, fixture.eps - Math.fround(fixture.eps), true);
          view.setUint32(16, fixture.tokens, true); view.setFloat32(20, 1, true);
          device.queue.writeBuffer(uniform, 0, uniformBytes);
          const input = alloc(fixture.input.length * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
          const residual = alloc(fixture.residual.length * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
          const weight = alloc(fixture.weight.length * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
          device.queue.writeBuffer(input, 0, new Float32Array(fixture.input));
          device.queue.writeBuffer(residual, 0, new Float32Array(fixture.residual));
          device.queue.writeBuffer(weight, 0, new Float32Array(fixture.weight));
          const bytes = fixture.input.length * 4;
          const output = alloc(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
          const prenorm = alloc(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
          const staging = alloc(bytes * 2, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
          const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [uniform, input, weight, output, residual, prenorm].map((buffer, binding) => ({ binding, resource: { buffer } })) });
          const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
          pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(fixture.tokens); pass.end();
          encoder.copyBufferToBuffer(output, 0, staging, 0, bytes); encoder.copyBufferToBuffer(prenorm, 0, staging, bytes, bytes);
          device.queue.submit([encoder.finish()]); await staging.mapAsync(GPUMapMode.READ); mapped.push(staging);
          results.push(Array.from(new Float32Array(staging.getMappedRange().slice(0))));
        } finally { for (const b of mapped) b.unmap(); await device.queue.onSubmittedWorkDone(); for (const b of owned) b.destroy(); }
      }
      return { results, errors, adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture } };
    } finally { await device.queue.onSubmittedWorkDone(); device.destroy(); }
  })({ source, cases });
  receipt.adapter = physical.adapter; receipt.errors = physical.errors;
  for (let c = 0; c < cases.length; c++) {
    const f = cases[c], values = physical.results[c]; let maxError = 0, failures = 0;
    for (let token = 0; token < f.tokens; token++) {
      const inputs = [], residuals = [];
      for (let i = 0; i < f.hidden; i++) {
        const index = (token * f.hidden + i) * 2;
        const residual = f.residual[index] + f.residual[index + 1]; residuals.push(residual);
        inputs.push(f.input[index] + f.input[index + 1] + (f.pre ? residual : 0));
      }
      const inverse = 1 / Math.sqrt(inputs.reduce((sum, value) => sum + value * value, 0) / f.hidden + f.eps);
      for (let i = 0; i < f.hidden; i++) {
        const index = (token * f.hidden + i) * 2;
        const expected = inputs[i] * inverse * (f.weight[i] + (f.offset ? 1 : 0)) + (f.post ? residuals[i] : 0);
        const error = Math.abs(values[index] + values[index + 1] - expected) / Math.max(1, Math.abs(expected));
        maxError = Math.max(maxError, error); if (!Number.isFinite(error) || error > receipt.tolerance) failures++;
        if (f.pre && values[index + f.input.length] + values[index + 1 + f.input.length] !== inputs[i]) {
          const err = Math.abs(values[index + f.input.length] + values[index + 1 + f.input.length] - inputs[i]);
          if (err > 2e-14) failures++;
        }
      }
    }
    receipt.cases.push({ hidden: f.hidden, pre: f.pre, maxError, failures });
  }
  receipt.numericalPassed = receipt.cases.every(row => row.failures === 0) && receipt.errors.length === 0;

assert(receipt.numericalPassed);receipt.operationSettled=true;persist();})();await bounded(operation,300000,'operation');
}catch(error){aborted=true;receipt.error=error.stack;process.exitCode=1;persist();}
finally{try{await bounded((async()=>{if(!receipt.operationSettled)await operation?.catch(()=>{});await host.close();receipt.hostClosed=true;})(),10000,'cleanup');}catch(error){receipt.cleanupError=error.message;process.exitCode=1;}receipt.passed=receipt.operationSettled===true&&receipt.hostClosed===true&&!receipt.error;mark('finished',{passed:receipt.passed});}
