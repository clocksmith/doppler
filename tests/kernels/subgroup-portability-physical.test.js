import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, getKernelCapabilities, destroyDevice } from '../../src/gpu/device.js';
import { runSubgroupPortabilityCases } from '../helpers/subgroup-portability.js';

const probe = await probeNodeGPU();
if (!probe.ready) throw new Error(`Subgroup operator verification requires WebGPU: ${probe.reason}`);
const device = getDevice();
const caps = getKernelCapabilities();
assert.doesNotMatch(JSON.stringify(caps.adapterInfo), /swiftshader|llvmpipe|software/i);
assert(caps.hasSubgroups && caps.wgslLanguageFeatures.includes('subgroup_id'));

const stats = await readFile(new URL('../../src/gpu/kernels/rmsnorm_stats_subgroups.wgsl', import.meta.url), 'utf8');
const portableStats = await readFile(new URL('../../src/gpu/kernels/rmsnorm_stats.wgsl', import.meta.url), 'utf8');
const attention = await readFile(new URL('../../src/gpu/kernels/attention_decode_subgroup.wgsl', import.meta.url), 'utf8');
try {
  const counts = await runSubgroupPortabilityCases(device, { stats, portableStats, attention });
  console.log(JSON.stringify({ test: 'subgroup-portability-physical', passed: true, ...counts,
    adapter: caps.adapterInfo, languageFeatures: caps.wgslLanguageFeatures,
    evidence: 'Local Node WebGPU operator parity, original and instrumented data-index permutations; no model qualification or other-hardware claim.' }));
} finally { destroyDevice(); }
