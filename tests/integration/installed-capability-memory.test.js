import assert from 'node:assert/strict';
import vm from 'node:vm';
import { installCapabilityMemoryProbe } from '../fixtures/installed-capability-memory.js';

const context = vm.createContext({ setTimeout });
vm.runInContext(`
  class GPUBuffer { constructor(size) { this.size = size; } destroy() {} }
  class GPUDevice {
    constructor() { this.lost = new Promise(resolve => { this.lose = resolve; });
      this.queue = { onSubmittedWorkDone: async () => {} }; }
    createBuffer({ size }) { if (size < 0) throw new RangeError('Injected GPU allocation failure'); return new GPUBuffer(size); }
    destroy() {}
  }
  const performance = {};
  (${installCapabilityMemoryProbe.toString()})();
  const first = new GPUDevice(), second = new GPUDevice();
  const a = first.createBuffer({ size: 100, label: 'session' }), b = second.createBuffer({ size: 200, label: 'session' });
`, context);
const metrics = () => JSON.parse(vm.runInContext('JSON.stringify(readCapabilityMemory().gpuBuffers)', context));
const labels = () => JSON.parse(vm.runInContext('JSON.stringify(readCapabilityMemory().liveBufferLabels)', context));
assert.equal(metrics().liveBytes, 300);
assert.deepEqual(labels(), [{ label: 'session', bytes: 300, count: 2 }]);
assert.throws(() => vm.runInContext('first.createBuffer({ size: -1 })', context), /Injected GPU allocation failure/);
assert.equal(metrics().failedAllocations, 1);
vm.runInContext('a.destroy(); a.destroy()', context);
assert.equal(metrics().liveBytes, 200);
assert.deepEqual(labels(), [{ label: 'session', bytes: 200, count: 1 }]);
vm.runInContext('first.destroy()', context);
assert.equal(metrics().liveBytes, 200, 'Device cleanup must not affect another device');
vm.runInContext('second.lose()', context);
await Promise.resolve();
vm.runInContext('b.destroy(); second.destroy()', context);
assert.equal(metrics().liveBytes, 0);
assert.equal(metrics().liveCount, 0);
assert.equal(metrics().destroyedBytes, 300);
assert.equal(metrics().peakLiveBytes, 300);
assert.deepEqual(labels(), [], 'observation metadata does not accumulate closed allocation labels');
assert.deepEqual(JSON.parse(JSON.stringify(await vm.runInContext('settleCapabilityGPU()', context))), []);
console.log('installed-capability-memory: passed (instrumentation doubles, not physical evidence)');
