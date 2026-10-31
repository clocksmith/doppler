import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, getKernelCapabilities, destroyDevice } from '../../src/gpu/device.js';
import { getKernelConfig } from '../../src/gpu/kernels/kernel-configs.js';
const probe = await probeNodeGPU({ installFileFetchShim: true });
assert.ok(probe.ready, probe.reason);
const device = getDevice();
const adapter = getKernelCapabilities().adapterInfo;
assert.match(JSON.stringify(adapter), /software|swiftshader|llvmpipe|lavapipe/i,
  'This receipt is only for software compilation; use the physical numerical test on hardware.');
try {
  for (const [operation, variant, constants] of [
    ['matmul', 'q4_0_cooperative', {}], ['matmul', 'q8_0_cooperative', {}],
    ['moe_shared_slot_gu', 'native_f32', { TOP_K: 8, ROUTED_FORMAT: 0, ROUTED_BYTES: 18, SHARED_FORMAT: 4, SHARED_BYTES: 22 }],
    ['moe_shared_slot_down', 'native_f32', { TOP_K: 8, ROUTED_FORMAT: 1, ROUTED_BYTES: 20, SHARED_FORMAT: 4, SHARED_BYTES: 22 }],
  ]) {
    const config = getKernelConfig(operation, variant);
    const code = readFileSync(new URL(`../../src/gpu/kernels/${config.shaderFile}`, import.meta.url), 'utf8');
    device.pushErrorScope('validation');
    const module = device.createShaderModule({ code });
    const info = await module.getCompilationInfo();
    assert.deepEqual(info.messages.filter(m => m.type === 'error').map(m => m.message), []);
    await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: config.entryPoint,
      constants: { ...config.wgslOverrides, ...constants } } });
    assert.equal(await device.popErrorScope(), null);
    console.log(JSON.stringify({ operation, variant, compiled: true, adapter, numericalQualification: false }));
  }
} finally { destroyDevice(); }
