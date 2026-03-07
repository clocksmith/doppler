import assert from 'node:assert/strict';

import { snapshotTensor } from '../../src/debug/tensor.js';
import { setGPUDevice } from '../../src/debug/config.js';

setGPUDevice(null);

const result = await snapshotTensor({ size: 16 }, [4], 'f32');

assert.equal(result.ok, false);
assert.match(String(result.error || ''), /GPU device not initialized/);
assert.deepEqual(result.shape, [4]);
assert.equal(result.sample.length, 0);

console.log('debug-tensor-contract.test: ok');
