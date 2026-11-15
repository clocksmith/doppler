import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { acquireBuffer, uploadData, readBuffer, releaseBuffer } from '../../src/memory/buffer-pool.js';
import { runRouteRemap } from '../../src/gpu/kernels/moe-route.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) throw new Error(`Route remap oracle needs physical WebGPU: ${probe.reason}`);
const errors = [];
getDevice().addEventListener('uncapturederror', event => errors.push(event.error.message));
try {
  for (const count of [1, 10, 257]) {
    const map = new Uint32Array([2, 0xffffffff, 0, 1, 99]);
    const ids = Uint32Array.from({ length: count }, (_, i) => [3, 0, 2, 3, 1, 4, 5][i % 7]);
    const expected = Array.from(ids, id => id < map.length && map[id] < 3 ? map[id] : 0xffffffff);
    const source = acquireBuffer(ids.byteLength), table = acquireBuffer(map.byteLength);
    uploadData(source, ids); uploadData(table, map);
    let output;
    try {
      output = await runRouteRemap(source, table, count, map.length, 3);
      assert.deepEqual(Array.from(new Uint32Array(await readBuffer(output, count * 4))), expected);
      await assert.rejects(runRouteRemap(source, table, count, table.size / 4 + 1, 3), /slot map/);
    } finally {
      await getDevice().queue.onSubmittedWorkDone();
      for (const buffer of [source, table, output]) if (buffer) releaseBuffer(buffer);
    }
  }
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ cases: 3, exact: true, preservesOrderAndDuplicates: true, invalidSentinel: true }));
} finally { destroyDevice(); }
