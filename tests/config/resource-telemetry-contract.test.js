import assert from 'node:assert/strict';
import {
  parseResourceTelemetryMode,
  parseRocmSmiJson,
  summarizeResourceTelemetrySamples,
} from '../../tools/resource-telemetry.js';

assert.equal(parseResourceTelemetryMode('on'), true);
assert.equal(parseResourceTelemetryMode('off'), false);
assert.throws(() => parseResourceTelemetryMode('maybe'), /--resource-telemetry must be one of/);

const rocm = parseRocmSmiJson(JSON.stringify({
  card0: {
    'GPU use (%)': '75',
    'GPU Memory Allocated (VRAM%)': '83',
    'VRAM Total Memory (B)': '536870912',
    'VRAM Total Used Memory (B)': '447135744',
    'Current Socket Graphics Package Power (W)': '42.5',
  },
}));
assert.equal(rocm.provider, 'rocm-smi');
assert.equal(rocm.deviceCount, 1);
assert.equal(rocm.utilizationPercent, 75);
assert.equal(rocm.memoryUsedBytes, 447135744);
assert.equal(rocm.memoryTotalBytes, 536870912);
assert.equal(rocm.powerWatts, 42.5);

const summary = summarizeResourceTelemetrySamples({
  pid: 123,
  label: 'unit',
  intervalMs: 100,
  includeSamples: false,
  startedAtMs: 1000,
  endedAtMs: 1200,
  sources: {
    procfs: true,
    meminfo: true,
    rocmSmi: true,
  },
  samples: [
    {
      timestamp: '2026-07-07T00:00:01.000Z',
      timestampMs: 1000,
      process: {
        processCount: 1,
        rssBytes: 1024,
        hwmBytes: 1024,
        userMs: 10,
        systemMs: 5,
        totalCpuMs: 15,
        cpuPercent: null,
      },
      system: {
        ramTotalBytes: 4096,
        ramAvailableBytes: 3072,
        ramFreeBytes: 2048,
        ramUsedBytes: 1024,
      },
      gpu: {
        provider: 'rocm-smi',
        deviceCount: 1,
        utilizationPercent: 10,
        memoryUsedBytes: 2048,
        memoryTotalBytes: 8192,
        memoryUsedPercent: 25,
        powerWatts: 20,
      },
      unavailableReasons: [],
    },
    {
      timestamp: '2026-07-07T00:00:01.100Z',
      timestampMs: 1100,
      process: {
        processCount: 2,
        rssBytes: 2048,
        hwmBytes: 2048,
        userMs: 30,
        systemMs: 10,
        totalCpuMs: 40,
        cpuPercent: 25,
      },
      system: {
        ramTotalBytes: 4096,
        ramAvailableBytes: 2048,
        ramFreeBytes: 1024,
        ramUsedBytes: 2048,
      },
      gpu: {
        provider: 'rocm-smi',
        deviceCount: 1,
        utilizationPercent: 30,
        memoryUsedBytes: 4096,
        memoryTotalBytes: 8192,
        memoryUsedPercent: 50,
        powerWatts: 30,
      },
      unavailableReasons: ['unit-warning'],
    },
  ],
});

assert.equal(summary.schemaVersion, 1);
assert.equal(summary.label, 'unit');
assert.equal(summary.durationMs, 200);
assert.equal(summary.process.rssBytes.max, 2048);
assert.equal(summary.process.cpuPercent.mean, 25);
assert.equal(summary.system.ramUsedBytes.max, 2048);
assert.equal(summary.gpu.utilizationPercent.mean, 20);
assert.equal(summary.gpu.memoryUsedBytes.max, 4096);
assert.deepEqual(summary.sampling.unavailableReasons, ['unit-warning']);
assert.equal(Object.hasOwn(summary, 'samples'), false);

console.log('resource-telemetry-contract.test: ok');
