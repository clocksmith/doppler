import assert from 'node:assert/strict';

import {
  currentLogLevel,
  LOG_LEVELS,
  getTrace,
  setLogLevel,
  setSilentMode,
  setTrace,
  traceLayerFilter,
  traceMaxDecodeSteps,
  traceBreakOnAnomaly,
} from '../../src/debug/config.js';

const originalWarn = console.warn;

{
  setLogLevel('warn');
  assert.equal(currentLogLevel, LOG_LEVELS.WARN);
  assert.throws(
    () => setLogLevel('inf0'),
    /Unknown log level "inf0"/
  );
  assert.equal(currentLogLevel, LOG_LEVELS.WARN);
}

{
  assert.throws(
    () => setTrace('perf,typo'),
    /Unknown trace category "typo"/
  );

  assert.throws(
    () => setTrace(['perf'], { layers: ['x'] }),
    /setTrace\(options\)\.layers\[0\] must be a non-negative integer/
  );

  setTrace(['perf', 'kv'], {
    layers: [1, 2],
    maxDecodeSteps: 4,
    breakOnAnomaly: true,
  });
  assert.deepEqual(getTrace(), ['perf', 'kv']);
  assert.deepEqual(traceLayerFilter, [1, 2]);
  assert.equal(traceMaxDecodeSteps, 4);
  assert.equal(traceBreakOnAnomaly, true);
  setTrace(false);
}

{
  let warnCount = 0;
  console.warn = () => {
    warnCount += 1;
  };
  setSilentMode(true);
  console.warn('should be silenced');
  setSilentMode(false);
  assert.equal(warnCount, 0);
}

console.warn = originalWarn;

console.log('debug-config-contract.test: ok');
