import assert from 'node:assert/strict';
import { runRepeatChannels, recordRepeatChannels } from '../../src/gpu/kernels/repeat_channels.js';

const input = { dtype: 'f32', buffer: { size: 4 }, shape: [1, 1, 1] };
const valid = { inChannels: 1, height: 1, width: 1, repeats: 1 };
for (const name of Object.keys(valid)) {
  for (const value of [0, -1, 0.5, NaN, Infinity, undefined]) {
    const options = { ...valid, [name]: value };
    await assert.rejects(runRepeatChannels(input, options), /positive integer/);
    await assert.rejects(recordRepeatChannels({}, input, options), /positive integer/);
  }
}
await assert.rejects(runRepeatChannels(input, { ...valid, inChannels: 0xFFFFFFFF, repeats: 2 }), /fit a u32/);
await assert.rejects(recordRepeatChannels({}, input, { ...valid, height: 65536, width: 65536 }), /fit a u32/);
console.log('repeat-channels-contract.test: ok');
