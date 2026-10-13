import assert from 'node:assert/strict';
import { readFileSync, appendFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
const [root, shader, log] = process.argv.slice(2);
const hash = value => createHash('sha256').update(value).digest('hex');
const mark = (phase, fields = {}) => appendFileSync(log, JSON.stringify({ phase,
  at: new Date().toISOString(), pid: process.pid, rss: process.memoryUsage().rss, ...fields }) + '\n');
const cg = readFileSync('/proc/self/cgroup', 'utf8').trim().split(':').at(-1);
// A scope inherits the SSH session's render group; its memory controller is owned by this user.
assert.ok(cg.endsWith('/reploid-log-alone-20261013.scope'));
writeFileSync('/sys/fs/cgroup' + cg + '/memory.oom.group', '1');
for (const [key, expected] of Object.entries({ 'memory.high': '6442450944',
  'memory.max': '8589934592', 'memory.swap.max': '0', 'memory.oom.group': '1' })) {
  assert.equal(readFileSync('/sys/fs/cgroup' + cg + '/' + key, 'utf8').trim(), expected);
}
const manifest = JSON.parse(readFileSync(shader, 'utf8'));
assert.equal(hash(readFileSync(resolve(root, 'node_modules/webgpu/dist/linux-x64.dawn.node'))),
  'fbfe599055b469963e4c9bfc900a702843db962dcaba0799b3c81ae33fa43db1');
mark('limits-and-inputs-verified', { cgroup: cg, manifest });
const { createNativeGpuHost } = await import(pathToFileURL(resolve(root, 'server/native-participant.js')));
const host = createNativeGpuHost(() => import(pathToFileURL(resolve(root, 'node_modules/webgpu/index.js'))));
try {
  await host.open(); mark('host-open');
  const adapter = await navigator.gpu.requestAdapter();
  assert.ok(adapter && !adapter.info.isFallbackAdapter);
  mark('adapter', { vendor: adapter.info.vendor, architecture: adapter.info.architecture,
    device: adapter.info.device, description: adapter.info.description });
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 9 } });
  const { setDevice } = await import(pathToFileURL(resolve(root, 'node_modules/doppler-gpu/src/gpu/device.js')));
  setDevice(device, { platformConfig: null });
  device.lost.then(info => mark('device-lost', { reason: info.reason, message: info.message }));
  device.addEventListener('uncapturederror', event => mark('gpu-error', { message: event.error.message }));
  for (const item of manifest.shaders) {
  const source = readFileSync(resolve(shader, '..', item.file), 'utf8');
  assert.equal(hash(source), item.sha256);
  mark('shader-module-start', { file: item.file, sha256: item.sha256 });
  const module = device.createShaderModule({ code: source });
  const errors = (await module.getCompilationInfo()).messages.filter(row => row.type === 'error');
  mark('shader-module-complete', { errors }); assert.equal(errors.length, 0);
  mark('pipeline-start', { file: item.file });
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  mark('pipeline-complete', { file: item.file });
  const { inputs } = JSON.parse(readFileSync(resolve(shader, '..', 'inputs.json'), 'utf8'));
  const count = inputs.length / 2;
  const bytes = count * 8;
  const owned = []; let staging;
  const alloc = (size, usage) => { assert(size <= 1420000000); const b = device.createBuffer({size,usage}); owned.push(b); return b; };
  try {
    const input = alloc(inputs.length*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
    const output = alloc(bytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
    staging = alloc(bytes,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
    device.queue.writeBuffer(input,0,new Float32Array(inputs));
    const bindings = device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[input,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();
    pass.setPipeline(pipeline);pass.setBindGroup(0,bindings);pass.dispatchWorkgroups(Math.ceil(count/64));pass.end();
    encoder.copyBufferToBuffer(output,0,staging,0,bytes);device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    mark('readback',{ count, values:Array.from(new Float32Array(staging.getMappedRange())).map(v=>Number.isFinite(v)?v:String(v)) });
  } finally {
    if(staging?.mapState==='mapped') staging.unmap();
    await device.queue.onSubmittedWorkDone(); for(const b of owned) b.destroy();
  }

  }
} catch (error) {
  mark('error', { message: error.stack }); process.exitCode = 1;
} finally {
  mark('host-close-start');
  try { await host.close(); mark('host-close-complete'); }
  catch (error) { mark('host-close-failed', { message: error.message }); process.exitCode = 1; }
}
