import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { packedFixture, decodePacked, projectReference } from '../helpers/q4_0-fixture.js';

// Direct shader stress at the device's largest bounded workgroup. This is not
// a registered model geometry. Report whether this GPU actually needed a fold.
const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) {
  console.log(`gguf-column-wave-fold-gpu.test: skipped (${probe.reason})`);
  process.exit(0);
}
const { getDevice, getKernelCapabilities, destroyDevice } = await import('../../src/gpu/device.js');
if (!getKernelCapabilities()?.hasSubgroups) {
  destroyDevice();
  console.log('gguf-column-wave-fold-gpu.test: skipped (no subgroups)');
  process.exit(0);
}
const { acquireBuffer, releaseBuffer, uploadData, readBuffer } = await import('../../src/memory/buffer-pool.js');
const { createCommandRecorder } = await import('../../src/gpu/command-recorder.js');
const { getKernelConfig } = await import('../../src/gpu/kernels/kernel-configs.js');
const { getShaderModule } = await import('../../src/gpu/kernels/shader-cache.js');
const { createKernelUniformBuffer } = await import('../../src/gpu/kernels/uniform-utils.js');
const device = getDevice(), pooled = [], direct = [], errors = [];
const width = Math.min(1024, device.limits.maxComputeInvocationsPerWorkgroup, device.limits.maxComputeWorkgroupSizeX);
const upload = data => {
  const buffer = acquireBuffer(Math.ceil(data.byteLength / 4) * 4);
  uploadData(buffer, data); pooled.push(buffer); return buffer;
};
device.addEventListener('uncapturederror', event => errors.push(event.error.message));
try {
  const waveInfo = upload(new Uint32Array(2));
  const infoModule = device.createShaderModule({ code: `enable subgroups; requires subgroup_id;
    @group(0) @binding(0) var<storage, read_write> info: array<u32>;
    @compute @workgroup_size(${width}) fn main(@builtin(local_invocation_index) t: u32,
      @builtin(subgroup_size) size: u32, @builtin(num_subgroups) count: u32) {
      if (t == 0u) { info[0] = size; info[1] = count; }
    }` });
  const infoPipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: infoModule, entryPoint: 'main' } });
  const infoBind = device.createBindGroup({ layout: infoPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: waveInfo } }] });
  const infoRec = createCommandRecorder('wave_fold_info');
  infoRec.recordDispatch(infoPipeline, infoBind, [1, 1, 1]); await infoRec.submitAndWait();
  const info = new Uint32Array(await readBuffer(waveInfo, 8));
  assert.equal(info[0] * info[1], width);
  const M = 4, N = 7, K = 8192, prefix = 4;
  const input = Float32Array.from({ length: M * K }, (_, i) => Math.sin(i * 0.017) * 0.125);
  const A = upload(input);
  for (const dtype of ['q4_0', 'q8_0', 'q6k']) {
    const bytes = packedFixture(dtype, N, K), weights = upload(bytes);
    const expected = projectReference(input, decodePacked(dtype, bytes, N, K), M, N, K);
    const output = upload(new Float32Array(M * N + 12).fill(-12345));
    const config = getKernelConfig('matmul', `${dtype}_pair_columns_hoisted`);
    const module = await getShaderModule(device, config.shaderFile, 'wave_fold_stress');
    const constants = { ...config.wgslOverrides, WORKGROUP_SIZE: width };
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: config.entryPoint, constants } });
    const uniform = createKernelUniformBuffer('wave_fold_uniform', config,
      { M, N, K, alpha: 1, blocks: K / config.wgslOverrides.BLOCK_VALUES, a_offset: 0, b_row: 0, c_offset: prefix }, null, device);
    direct.push(uniform);
    const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
      entries: [uniform, A, weights, output].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const rec = createCommandRecorder('wave_fold_reference');
    rec.recordDispatch(pipeline, bind, [Math.ceil(N / 4), 1, 1]); await rec.submitAndWait();
    const actual = new Float32Array(await readBuffer(output, (M * N + 12) * 4));
    let maxAbs = 0;
    for (let i = 0; i < expected.length; i++) {
      const delta = Math.abs(actual[prefix + i] - expected[i]); maxAbs = Math.max(maxAbs, delta);
      assert.ok(Number.isFinite(actual[prefix + i]) && delta <= 2e-4 + 2e-5 * Math.abs(expected[i]));
    }
    assert.ok(actual.subarray(0, prefix).every(value => value === -12345));
    assert.ok(actual.subarray(prefix + M * N).every(value => value === -12345));
    console.log(JSON.stringify({ dtype, width, subgroupSize: info[0], waves: info[1], exercisedFold: info[1] > 16, maxAbs }));
  }
  assert.deepEqual(errors, []);
} finally {
  await device.queue.onSubmittedWorkDone();
  for (const buffer of pooled) releaseBuffer(buffer);
  for (const buffer of direct) buffer.destroy();
  destroyDevice();
}
