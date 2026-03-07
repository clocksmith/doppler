import assert from 'node:assert/strict';

const { MemoryMonitor, MemoryTimeSeries } = await import('../../src/loader/memory-monitor.js');

const originalSetInterval = globalThis.setInterval;
const originalClearInterval = globalThis.clearInterval;

let nextId = 1;
const cleared = [];
globalThis.setInterval = ((fn, ms) => ({ id: nextId++, fn, ms }));
globalThis.clearInterval = ((handle) => {
  cleared.push(handle?.id ?? handle);
});

try {
  {
    const monitor = new MemoryMonitor(1000, false);
    const getState = () => ({
      shardCacheBytes: 0,
      shardCount: 0,
      layerCount: 0,
      gpuBufferCount: 0,
    });

    monitor.start(getState);
    monitor.start(getState);
    monitor.stop('done', getState);

    assert.deepEqual(cleared.slice(0, 2), [1, 2]);
  }

  {
    const series = new MemoryTimeSeries(1000);
    series.start();
    series.start();
    series.stop();

    assert.deepEqual(cleared.slice(2), [3, 4]);
  }
} finally {
  globalThis.setInterval = originalSetInterval;
  globalThis.clearInterval = originalClearInterval;
}

console.log('memory-monitor.test: ok');
