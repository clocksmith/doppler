import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { float32ToFloat16 } from '../../src/converter/quantizer.js';
import { getKernelConfig } from '../../src/config/kernel-registry-contract.js';
import { writeUniformsFromObject } from '../../src/gpu/kernels/uniform-utils.js';

// Explicit opt-in paired physical probe: both lanes retain independent state,
// execute the same recurrent steps, and compare every state/output byte.
const config = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
assert.equal(config.schema, 'doppler.fused-recurrent-profile/v1');
for (const key of ['warmupPairs', 'measuredPairs', 'heads', 'headDim', 'workgroupSize', 'steps']) {
  assert(Number.isSafeInteger(config[key]) && config[key] > 0, key);
}
const probe = await probeNodeGPU();
assert(probe.ready, probe.reason);
const device = getDevice();
assert(device.features.has('timestamp-query'), 'Physical timestamps are required');
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
    for (const components of config.components) {
      assert([1, 2].includes(components));
      const kernel = getKernelConfig('gated_delta_fused_decode', variant);
      const pipelines = {}, sources = {};
      for (const lane of ['baseline', 'candidate']) {
        const source = await fs.readFile(path.join(config[lane], 'src/gpu/kernels', kernel.shaderFile), 'utf8');
        sources[lane] = hash(source);
        const module = device.createShaderModule({ code: source });
        const info = await module.getCompilationInfo();
        assert.deepEqual(info.messages.filter(message => message.type === 'error'), []);
        pipelines[lane] = await device.createComputePipelineAsync({ layout: 'auto', compute: {
          module, entryPoint: kernel.entryPoint,
          constants: { WORKGROUP_SIZE: config.workgroupSize, RECURRENT_STATE_COMPONENTS: components },
        } });
      }
      const { heads, headDim } = config;
      const valueDim = heads * headDim;
      const convDim = valueDim * 3;
      const input = (length, offset) => Float32Array.from({ length }, (_, i) => Math.sin(i * 0.31 + offset) * 0.7);
      const encode = values => variant.includes('f16') ? Uint16Array.from(values, float32ToFloat16) : values;
      const values = {
        num_tokens: 1, conv_dim: convDim, conv_kernel_size: 4,
        num_v_heads: heads, num_k_heads: heads, head_k_dim: headDim, head_v_dim: headDim,
        q_size: valueDim, k_size: valueDim, value_dim: valueDim, q_rep: 1, norm_mode: 0,
        rms_norm_eps: 1e-6, qk_l2norm_eps: 1e-6, packed_flags: 1, b_proj_offset_elements: heads,
      };
      const uniform = new ArrayBuffer(kernel.uniforms.size);
      writeUniformsFromObject(new DataView(uniform), kernel, values);
      const arrays = [encode(input(convDim, 1)), encode(input(valueDim, 2)), encode(input(heads * 2, 3)),
        input(convDim * 4, 4), input(heads, 5), input(heads, 6), Float32Array.from({ length: headDim }, () => 1)];
      const common = [buffer(uniform.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, uniform),
        ...arrays.map(array => buffer(array.byteLength, storage, array))];
      const initial = input(heads * headDim * headDim * components, 7);
      if (components === 2) for (let i = 1; i < initial.length; i += 2) initial[i] *= 1e-8;
      const initialConv = input(convDim * 4, 8);
      const states = {}, convStates = {}, outputs = {}, bindings = {};
      const outputBytes = valueDim * 4;
      for (const lane of ['baseline', 'candidate']) {
        states[lane] = buffer(initial.byteLength, storage);
        convStates[lane] = buffer(initialConv.byteLength, storage);
        outputs[lane] = buffer(outputBytes, storage);
        const resources = [...common.slice(0, 5), convStates[lane], ...common.slice(5), states[lane], outputs[lane]];
        bindings[lane] = device.createBindGroup({ layout: pipelines[lane].getBindGroupLayout(0),
          entries: resources.map((entry, binding) => ({ binding, resource: { buffer: entry } })) });
      }
      const resolved = buffer(16, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC);
      const staging = buffer(16 + initial.byteLength + initialConv.byteLength + outputBytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      const row = { variant, components, sources, initialHash: hash(initial), pairs: [] };
      report.cases.push(row);
      for (let pair = -config.warmupPairs; pair < config.measuredPairs; pair++) {
        const order = pair % 2 ? ['candidate', 'baseline'] : ['baseline', 'candidate'];
        const result = { pair, order };
        for (const lane of order) {
          device.queue.writeBuffer(states[lane], 0, initial);
          device.queue.writeBuffer(convStates[lane], 0, initialConv);
          const encoder = device.createCommandEncoder();
          const pass = encoder.beginComputePass({ timestampWrites: {
            querySet: query, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1,
          } });
          pass.setPipeline(pipelines[lane]);
          pass.setBindGroup(0, bindings[lane]);
          for (let step = 0; step < config.steps; step++) pass.dispatchWorkgroups(heads);
          pass.end();
          encoder.resolveQuerySet(query, 0, 2, resolved, 0);
          encoder.copyBufferToBuffer(resolved, 0, staging, 0, 16);
          encoder.copyBufferToBuffer(states[lane], 0, staging, 16, initial.byteLength);
          encoder.copyBufferToBuffer(convStates[lane], 0, staging, 16 + initial.byteLength, initialConv.byteLength);
          encoder.copyBufferToBuffer(outputs[lane], 0, staging, 16 + initial.byteLength + initialConv.byteLength, outputBytes);
          const started = performance.now();
          device.queue.submit([encoder.finish()]);
          await staging.mapAsync(GPUMapMode.READ);
          const wallMs = performance.now() - started;
          const mapped = staging.getMappedRange();
          const times = new BigUint64Array(mapped, 0, 2);
          assert(new Float32Array(mapped, 16).every(Number.isFinite));
          result[lane] = { gpuMs: Number(times[1] - times[0]) / 1e6, wallMs,
            stateHash: hash(new Uint8Array(mapped, 16, initial.byteLength)),
            convHash: hash(new Uint8Array(mapped, 16 + initial.byteLength, initialConv.byteLength)),
            outputHash: hash(new Uint8Array(mapped, 16 + initial.byteLength + initialConv.byteLength, outputBytes)) };
          staging.unmap();
        }
        row.pairs.push({ ...result, warmup: pair < 0 });
        for (const key of ['stateHash', 'convHash', 'outputHash']) assert.equal(result.baseline[key], result.candidate[key], key);
      }
      console.log(JSON.stringify({ variant, components, passed: true }));
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
