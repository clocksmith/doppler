import assert from 'node:assert/strict';

import { createCommandExecutor } from '../../src/client/runtime/command-executor.js';
import { createResourceBinder } from '../../src/client/runtime/resource-binder.js';
import { destroyDevice, getDevice } from '../../src/gpu/device.js';
import { probeNodeGPU } from '../helpers/gpu-probe.js';

const probe = await probeNodeGPU();
if (!probe.ready) {
  if (globalThis.__DOPPLER_REQUIRE_PHYSICAL_WEBGPU__ === true) {
    throw new Error(`Physical Node WebGPU is required: ${probe.reason}`);
  }
  console.log(`pack-command-executor-physical.test.js: skipped (${probe.reason})`);
  process.exit(0);
}

const device = getDevice();
const binder = createResourceBinder(device);
binder.bindSlots({
  kvCacheLayout: 'contiguous',
  bufferSlots: [{
    slotId: 'output',
    role: 'test-output',
    scope: 'transient',
    owner: 'runtime',
    usage: ['storage', 'copy-src'],
    size: { op: 'constant', bytes: 4 },
  }],
}, {});

const source = `
@group(0) @binding(0) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(1)
fn main() {
  output[0] = 42u;
}
`;
const modules = new Map([['write-42', {
  id: 'write-42',
  entry: 'main',
  sourceHash: 'sha256:physical-test',
  source,
}]]);
const executor = createCommandExecutor(device, binder);
const result = await executor.executePhase('prefill', [{
  id: 'write-output',
  kind: 'dispatch',
  moduleId: 'write-42',
  entry: 'main',
  bindings: [{ binding: 0, slotId: 'output' }],
  workgroups: [1, 1, 1],
  waitForCompletion: true,
}], { modules });

assert.equal(result.commandCount, 1);
const staging = device.createBuffer({
  label: 'doppler-pack:physical-readback',
  size: 4,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});
const encoder = device.createCommandEncoder({ label: 'doppler-pack:physical-readback' });
encoder.copyBufferToBuffer(binder.getSlot('output').buffer, 0, staging, 0, 4);
device.queue.submit([encoder.finish()]);
await staging.mapAsync(GPUMapMode.READ);
const observed = new Uint32Array(staging.getMappedRange().slice(0))[0];
staging.unmap();
assert.equal(observed, 42);

staging.destroy();
binder.releaseAll();
executor.clearPipelineCache();
destroyDevice();

console.log('pack-command-executor-physical.test.js: physical WebGPU dispatch/readback passed');
