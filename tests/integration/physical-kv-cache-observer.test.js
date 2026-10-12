import assert from 'node:assert/strict';
import { observePhysicalKVCache } from '../helpers/physical-kv-cache-observer.js';

globalThis.GPUBufferUsage = { COPY_DST: 1, MAP_READ: 2 };
globalThis.GPUMapMode = { READ: 1 };
class Buffer {
  constructor(data) { this.data = new Uint8Array(data); this.size = this.data.length; this.mapState = 'unmapped'; }
  async mapAsync() { if (this.rejectMap) throw new Error('injected mapping failure'); this.mapState = 'mapped'; }
  getMappedRange() { return this.data.buffer; }
  unmap() { this.mapState = 'unmapped'; }
  destroy() { this.destroyed = true; }
}
const sourceK = new Buffer([0, 60, 0, 64, 0, 66, 0, 68]);
const sourceV = new Buffer([0, 69, 0, 70, 0, 71, 0, 72]);
class Cache {
  constructor() {
    this.layout = 'contiguous'; this.kvDtype = 'f16'; this.numHeads = 1; this.headDim = 2;
    this.layers = [{ seqLen: 2 }];
  }
  getGPUBuffers() { return { keysGPU: sourceK, valuesGPU: sourceV, seqLen: 2 }; }
}
const staged = [];
let rejectMap = false;
const device = {
  createBuffer({ size }) { const buffer = new Buffer(size); buffer.rejectMap = rejectMap; staged.push(buffer); return buffer; },
  createCommandEncoder() {
    const commands = [];
    return { copyBufferToBuffer: (...args) => commands.push(args), finish: () => commands };
  },
  queue: { submit(commands) {
    for (const command of commands.flat()) {
      const [source, offset, target, targetOffset, bytes] = command;
      target.data.set(source.data.subarray(offset, offset + bytes), targetOffset);
    }
  } }
};
const original = Cache.prototype.getGPUBuffers;
const observer = observePhysicalKVCache({ KVCache: Cache, getDevice: () => device, layers: [0] });
const cache = new Cache();
try {
  cache.getGPUBuffers(0);
  const pending = observer.capture(0, 2);
  sourceK.data.fill(0); // A later write must not change the submitted snapshot.
  const rows = await pending;
  assert.equal(rows[0].values.k.dataBase64, btoa(String.fromCharCode(0, 60, 0, 64, 0, 66, 0, 68)));
  assert(staged.every(buffer => buffer.destroyed && buffer.mapState === 'unmapped'));
  assert(!sourceK.destroyed && !sourceV.destroyed);
  rejectMap = true;
  cache.getGPUBuffers(0);
  await assert.rejects(observer.capture(1, 2), /injected mapping failure/);
  assert(staged.every(buffer => buffer.destroyed));
  assert.throws(() => observer.capture(2, 3), /geometry/);
} finally { observer.restore(); }
assert.equal(Cache.prototype.getGPUBuffers, original);
console.log('Physical cache observer: queued snapshot isolation, mapping-failure cleanup and geometry rejection passed (mock GPU).');

rejectMap = false;
const mapped = observePhysicalKVCache({ KVCache: Cache, getDevice: () => device, layers: [3], allocationLayerIndices: [3] });
try {
  const cache = new Cache(); cache.numLayers = 1;
  cache.getGPUBuffers(0);
  const rows = await mapped.capture(0, 2);
  assert.equal(rows[0].layer, 3); assert.equal(rows[0].allocationLayer, 0);
  cache.numLayers = 2;
  assert.throws(() => cache.getGPUBuffers(0), /allocation mapping/);
} finally { mapped.restore(); }
