import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getKernelCapabilities, destroyDevice } from '../../src/gpu/device.js';
import { runNativePortFixtures } from '../helpers/native-port-fixtures.js';
const probe = await probeNodeGPU({ installFileFetchShim: true });
assert.ok(probe.ready, probe.reason);
const adapter = getKernelCapabilities().adapterInfo;
try {
  assert.match(JSON.stringify(adapter), /software|swiftshader|llvmpipe|lavapipe/i);
  console.log(JSON.stringify({ adapter, providerReceipt: probe.providerReceipt,
    scope: 'Software-only numerical smoke; no physical GPU, full-model, or performance qualification' }));
  await runNativePortFixtures();
} finally { destroyDevice(); }
