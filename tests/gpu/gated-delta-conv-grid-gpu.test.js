import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { checkConvGridReference } from '../helpers/conv-grid-reference.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { getKernelConfig } from '../../src/gpu/kernels/kernel-configs.js';

const probe = await probeNodeGPU();
if (!probe.ready) { console.log(`gated-delta-conv-grid-gpu: skipped (${probe.reason})`); process.exit(0); }
const device = getDevice(), errors = [];
device.addEventListener('uncapturederror', event => errors.push(event.error.message));
const sources = {
  serial: await readFile(new URL('../../src/gpu/kernels/gated_delta_conv.wgsl', import.meta.url), 'utf8'),
  parallel: await readFile(new URL('../../src/gpu/kernels/gated_delta_conv_prefill.wgsl', import.meta.url), 'utf8'),
  workgroupSize: getKernelConfig('gated_delta_conv', 'prefill_parallel').workgroupSize[0],
};
try {
  for (const shape of [
    { numTokens: 2, convDim: 513, window: 4 },
    { numTokens: 7, convDim: 257, window: 4, packedValueDim: 193 },
    { numTokens: 2047, convDim: 8192, window: 4 },
    { numTokens: 2048, convDim: 8192, window: 4 },
  ]) console.log(JSON.stringify(await checkConvGridReference(device, sources, shape)));
  assert.deepEqual(errors, []);
  console.log('gated-delta-conv-grid-gpu: exact serial GPU control, CPU samples and causal history passed');
} finally { destroyDevice(); }
