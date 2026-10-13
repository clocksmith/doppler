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
assert.ok(cg.endsWith('/reploid-recurrent-phases-20261013.scope'));
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
  await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  mark('pipeline-complete', { file: item.file });
  }
} catch (error) {
  mark('error', { message: error.stack }); process.exitCode = 1;
} finally {
  mark('host-close-start');
  try { await host.close(); mark('host-close-complete'); }
  catch (error) { mark('host-close-failed', { message: error.message }); process.exitCode = 1; }
}
