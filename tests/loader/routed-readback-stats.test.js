import assert from 'node:assert/strict';
import { DopplerLoader } from '../../src/loader/doppler-loader.js';
import { getRuntimeConfig, setRuntimeConfig } from '../../src/config/runtime.js';

const previous = getRuntimeConfig();
const loader = Object.create(DopplerLoader.prototype);
loader.routedReadbackStats = {};
try {
  setRuntimeConfig({ shared: { debug: { profiler: { enabled: false } } } });
  loader.recordRoutedReadback('decode', 7, 40);
  assert.deepEqual(loader.getRoutedReadbackStats().phases, {});
  setRuntimeConfig({ shared: { debug: { profiler: { enabled: true } } } });
  loader.recordRoutedReadback('prefill', 3, 80);
  loader.recordRoutedReadback('decode', 4, 40);
  loader.recordRoutedReadback('decode', 5, 40);
  loader.recordRoutedReadback(undefined, 1, 4);
  const snapshot = loader.getRoutedReadbackStats();
  assert.deepEqual(snapshot.phases, {
    prefill: { calls: 1, bytes: 80, wallMs: 3 },
    decode: { calls: 2, bytes: 80, wallMs: 9 },
    unspecified: { calls: 1, bytes: 4, wallMs: 1 },
  });
  assert.match(snapshot.scope, /not GPU routing compute/);
  snapshot.phases.decode.wallMs = 999;
  assert.equal(loader.getRoutedReadbackStats().phases.decode.wallMs, 9);
} finally { setRuntimeConfig(previous); }
console.log('routed readback stats: passed');
