import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { float32ToFloat16 } from '../../src/converter/quantizer.js';
import { getKernelConfig } from '../../src/config/kernel-registry-contract.js';
import { writeUniformsFromObject } from '../../src/gpu/kernels/uniform-utils.js';

// A direct operator comparison, with alternating order, independent state,
// identical input bytes, timestamps, and exact output/state checks per pair.
const config = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
assert.equal(config.schema, 'doppler.recurrent-reduction-profile/v1');
for (const field of ['warmupPairs', 'measuredPairs', 'heads', 'headDim', 'workgroupSize']) {
  assert(Number.isSafeInteger(config[field]) && config[field] > 0, field);
}
const probe = await probeNodeGPU();
assert(probe.ready, probe.reason);
const device = getDevice();
assert(device.features.has('timestamp-query'), 'Physical GPU timestamps are required');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const report = { config, provider: probe.providerReceipt, startedAt: new Date().toISOString(), cases: [], passed: false };
const owned = [];
function buffer(size, usage, data) {
  const result = device.createBuffer({ size, usage });
  owned.push(result);
  if (data) device.queue.writeBuffer(result, 0, data);
  return result;
}
const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
const query = device.createQuerySet({ type: 'timestamp', count: 2 });
try {
  for (const variant of config.variants) {
    const kernel = getKernelConfig('gated_delta_recurrent', variant);
    const pipelines = {};
    const sources = {};
    for (const lane of ['baseline', 'candidate']) {
      const source = await fs.readFile(path.join(config[lane], 'src/gpu/kernels', kernel.shaderFile), 'utf8');
      sources[lane] = hash(source);
      const module = device.createShaderModule({ code: source });
      const info = await module.getCompilationInfo();
      assert.deepEqual(info.messages.filter(message => message.type === 'error'), []);
      pipelines[lane] = await device.createComputePipelineAsync({ layout: 'auto', compute: {
        module, entryPoint: kernel.entryPoint,
        constants: { WORKGROUP_SIZE: config.workgroupSize, RECURRENT_STATE_COMPONENTS: 1 },
      } });
    }
    for (const tokens of config.tokens) {
      assert(Number.isSafeInteger(tokens) && tokens > 0);
      const { heads, headDim } = config;
      const valueDim = heads * headDim;
      const input = (length, offset) => Float32Array.from({ length }, (_, i) => Math.sin(i * 0.31 + offset) * 0.7);
      const encode = values => variant.includes('f16') ? Uint16Array.from(values, float32ToFloat16) : values;
      const values = {
        num_tokens: tokens, conv_dim: valueDim * 3, conv_kernel_size: 4,
        num_v_heads: heads, num_k_heads: heads, head_k_dim: headDim, head_v_dim: headDim,
        q_size: valueDim, k_size: valueDim, value_dim: valueDim, q_rep: 1, norm_mode: 0,
        rms_norm_eps: 1e-6, qk_l2norm_eps: 1e-6, packed_flags: 0, b_proj_offset_elements: 0,
      };
      const uniform = new ArrayBuffer(kernel.uniforms.size);
      writeUniformsFromObject(new DataView(uniform), kernel, values);
      const arrays = [input(tokens * valueDim * 3, 1), encode(input(tokens * valueDim, 2)),
        encode(input(tokens * heads, 3)), encode(input(tokens * heads, 4)),
        input(heads, 5), input(heads, 6), Float32Array.from({ length: headDim }, () => 1)];
      const common = [buffer(uniform.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, uniform),
        ...arrays.map(array => buffer(array.byteLength, storage, array))];
      const initial = input(heads * headDim * headDim, 7);
      const states = {}, outputs = {}, bindings = {};
      const outputBytes = tokens * valueDim * 4;
      for (const lane of ['baseline', 'candidate']) {
        states[lane] = buffer(initial.byteLength, storage);
        outputs[lane] = buffer(outputBytes, storage);
        bindings[lane] = device.createBindGroup({ layout: pipelines[lane].getBindGroupLayout(0),
          entries: [...common, states[lane], outputs[lane]].map((entry, binding) => ({ binding, resource: { buffer: entry } })) });
      }
      const resolved = buffer(16, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC);
      const staging = buffer(16 + initial.byteLength + outputBytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      const row = { variant, tokens, sources, initialHash: hash(initial), pairs: [] };
      for (let pair = -config.warmupPairs; pair < config.measuredPairs; pair++) {
        const order = pair % 2 ? ['candidate', 'baseline'] : ['baseline', 'candidate'];
        const result = { pair, order };
        for (const lane of order) {
          device.queue.writeBuffer(states[lane], 0, initial);
          const encoder = device.createCommandEncoder();
          const pass = encoder.beginComputePass({ timestampWrites: {
            querySet: query, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1,
          } });
          pass.setPipeline(pipelines[lane]);
          pass.setBindGroup(0, bindings[lane]);
          pass.dispatchWorkgroups(heads);
          pass.end();
          encoder.resolveQuerySet(query, 0, 2, resolved, 0);
          encoder.copyBufferToBuffer(resolved, 0, staging, 0, 16);
          encoder.copyBufferToBuffer(states[lane], 0, staging, 16, initial.byteLength);
          encoder.copyBufferToBuffer(outputs[lane], 0, staging, 16 + initial.byteLength, outputBytes);
          const started = performance.now();
          device.queue.submit([encoder.finish()]);
          await staging.mapAsync(GPUMapMode.READ);
          const wallMs = performance.now() - started;
          const mapped = staging.getMappedRange();
          const times = new BigUint64Array(mapped, 0, 2);
          const floats = new Float32Array(mapped, 16);
          assert(floats.every(Number.isFinite), 'all state/output values must be finite');
          result[lane] = { gpuMs: Number(times[1] - times[0]) / 1e6, wallMs,
            stateHash: hash(new Uint8Array(mapped, 16, initial.byteLength)),
            outputHash: hash(new Uint8Array(mapped, 16 + initial.byteLength, outputBytes)) };
          staging.unmap();
        }
        assert.equal(result.baseline.stateHash, result.candidate.stateHash, 'state bytes must match exactly');
        assert.equal(result.baseline.outputHash, result.candidate.outputHash, 'output bytes must match exactly');
        if (pair >= 0) row.pairs.push(result);
      }
      report.cases.push(row);
      console.log(JSON.stringify({ variant, tokens, pairs: row.pairs.length, passed: true }));
      await device.queue.onSubmittedWorkDone();
      for (const resource of owned.splice(0)) resource.destroy();
    }
  }
  report.passed = true;
} finally {
  await device.queue.onSubmittedWorkDone();
  for (const resource of owned) resource.destroy();
  query.destroy();
  destroyDevice();
  await fs.writeFile(process.argv[3], `${JSON.stringify(report, null, 2)}\n`);
}
