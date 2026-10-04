import assert from 'node:assert/strict';
import { installDeviceMemoryAccounting, setDeviceMemoryBudget, getDeviceMemorySnapshot,
  markDeviceWeights } from '../../src/memory/device-budget.js';
let calls = 0, fail = false, lose;
const device = { lost: new Promise(resolve => { lose = resolve; }), createBuffer({ size }) {
  calls++; if (fail) throw new Error('native allocation failed');
  return { size, destroy() {} };
} };
installDeviceMemoryAccounting(device);
setDeviceMemoryBudget(device, 100);
const weight = device.createBuffer({size: 60, label: 'embedding'});
markDeviceWeights(device, [weight]);
const state = device.createBuffer({size: 32, label: 'kv_cache_keys_layer_0'});
assert.throws(() => device.createBuffer({size: 12}), error =>
  error.code === 'RESOURCE_EXHAUSTED' && /GPU memory budget exceeded/.test(error.message));
assert.equal(calls, 2, 'Over-budget allocations fail before the native allocator');
assert.throws(() => setDeviceMemoryBudget(device, 200), /Release existing/);
assert.throws(() => setDeviceMemoryBudget(device, undefined), /explicit null/);
state.destroy(); state.destroy();
assert.equal(getDeviceMemorySnapshot(device).liveBytes, 60);
assert.equal(getDeviceMemorySnapshot(device).categories.weights, 60);
fail = true;
assert.throws(() => device.createBuffer({size: 4}), /native allocation/);
assert.equal(getDeviceMemorySnapshot(device).liveBytes, 60);
fail = false;
const staging = device.createBuffer({size: 40, label: 'staging'});
assert.equal(getDeviceMemorySnapshot(device).peakBytes, 100);
weight.destroy(); staging.destroy();
assert.equal(getDeviceMemorySnapshot(device).liveBytes, 0);
setDeviceMemoryBudget(device, 50);
lose(); await Promise.resolve();
assert.throws(() => device.createBuffer({size: 4}), /lost GPU device/);
console.log('device-budget.test: ok');
