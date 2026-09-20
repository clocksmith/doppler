import assert from 'node:assert/strict';
import vm from 'node:vm';
import { installCapabilityMemoryProbe } from '../fixtures/installed-capability-memory.js';

const context = vm.createContext({});
vm.runInContext(`
  class GPUBuffer { constructor(size) { this.size = size; } destroy() {} }
  class GPUDevice {
    constructor() { this.lost = new Promise(resolve => { this.lose = resolve; }); }
    createBuffer({ size }) { if (size < 0) throw new RangeError('Injected GPU allocation failure'); return new GPUBuffer(size); }
    destroy() {}
  }
  const performance = {};
  (${installCapabilityMemoryProbe.toString()})();
  const first = new GPUDevice(), second = new GPUDevice();
  const a = first.createBuffer({ size: 100 }), b = second.createBuffer({ size: 200 });
`, context);
const metrics = () => JSON.parse(vm.runInContext('JSON.stringify(readCapabilityMemory().gpuBuffers)', context));
assert.equal(metrics().liveBytes, 300);
assert.throws(() => vm.runInContext('first.createBuffer({ size: -1 })', context), /Injected GPU allocation failure/);
assert.equal(metrics().failedAllocations, 1);
vm.runInContext('a.destroy(); a.destroy()', context);
assert.equal(metrics().liveBytes, 200);
vm.runInContext('first.destroy()', context);
assert.equal(metrics().liveBytes, 200, 'Device cleanup must not affect another device');
vm.runInContext('second.lose()', context);
await Promise.resolve();
vm.runInContext('b.destroy(); second.destroy()', context);
assert.equal(metrics().liveBytes, 0);
assert.equal(metrics().liveCount, 0);
assert.equal(metrics().destroyedBytes, 300);
assert.equal(metrics().peakLiveBytes, 300);
console.log('installed-capability-memory: passed (instrumentation doubles, not physical evidence)');
