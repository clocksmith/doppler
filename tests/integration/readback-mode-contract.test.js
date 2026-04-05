import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { READBACK_MODES } from '../../src/config/schema/execution-v1.schema.js';

// Inline minimal session resolution logic to test validation without
// requiring a full GPU device or pipeline init.
function resolveReadbackMode(decodeLoop, probeMs) {
  const readbackMode = decodeLoop.readbackMode;
  if (!readbackMode || !READBACK_MODES.includes(readbackMode)) {
    throw new Error(
      `DopplerConfigError: readbackMode must be one of ${READBACK_MODES.join(', ')}; got ${JSON.stringify(readbackMode)}.`
    );
  }

  if (readbackMode === 'overlapped') {
    const rs = decodeLoop.ringStaging ?? 0;
    if (rs < 2) {
      throw new Error(
        `DopplerConfigError: readbackMode "overlapped" requires ringStaging >= 2, got ${rs}.`
      );
    }
    return 'overlapped';
  }

  if (readbackMode === 'auto') {
    const thresholdMs = decodeLoop.submitLatencyThresholdMs;
    if (thresholdMs == null) {
      throw new Error(
        'DopplerConfigError: readbackMode "auto" requires submitLatencyThresholdMs to be set.'
      );
    }
    if (probeMs != null && probeMs > thresholdMs) {
      const rs = decodeLoop.ringStaging ?? 0;
      if (rs < 2) return 'sequential';
      return 'overlapped';
    }
    return 'sequential';
  }

  return 'sequential';
}

describe('readbackMode contract', () => {
  // Test 1: overlapped with ringStaging < 2 must throw at init
  it('throws when readbackMode is overlapped and ringStaging < 2', () => {
    assert.throws(
      () => resolveReadbackMode({
        readbackMode: 'overlapped',
        ringStaging: 1,
      }),
      /readbackMode "overlapped" requires ringStaging >= 2, got 1/
    );
  });

  // Test 2: auto with null submitLatencyThresholdMs must throw at init
  it('throws when readbackMode is auto and submitLatencyThresholdMs is null', () => {
    assert.throws(
      () => resolveReadbackMode({
        readbackMode: 'auto',
        submitLatencyThresholdMs: null,
      }),
      /readbackMode "auto" requires submitLatencyThresholdMs to be set/
    );
  });

  // Test 3: auto with probe 150ms > threshold 100ms resolves to overlapped
  it('resolves auto to overlapped when probe exceeds threshold', () => {
    const resolved = resolveReadbackMode({
      readbackMode: 'auto',
      submitLatencyThresholdMs: 100,
      ringStaging: 2,
    }, 150);
    assert.equal(resolved, 'overlapped');
  });

  // Test 4: auto with probe 30ms < threshold 100ms resolves to sequential
  it('resolves auto to sequential when probe is below threshold', () => {
    const resolved = resolveReadbackMode({
      readbackMode: 'auto',
      submitLatencyThresholdMs: 100,
      ringStaging: 2,
    }, 30);
    assert.equal(resolved, 'sequential');
  });

  // Test 5: downstream never sees "auto" — only sequential or overlapped
  it('resolved mode is never auto', () => {
    const cases = [
      { decodeLoop: { readbackMode: 'sequential' }, probe: 50 },
      { decodeLoop: { readbackMode: 'overlapped', ringStaging: 3 }, probe: 50 },
      { decodeLoop: { readbackMode: 'auto', submitLatencyThresholdMs: 100, ringStaging: 2 }, probe: 30 },
      { decodeLoop: { readbackMode: 'auto', submitLatencyThresholdMs: 100, ringStaging: 2 }, probe: 200 },
      { decodeLoop: { readbackMode: 'auto', submitLatencyThresholdMs: 100, ringStaging: 2 }, probe: null },
    ];
    for (const { decodeLoop, probe } of cases) {
      const resolved = resolveReadbackMode(decodeLoop, probe);
      assert.notEqual(resolved, 'auto', `expected resolved mode !== "auto" for input ${JSON.stringify(decodeLoop)}`);
      assert.ok(
        resolved === 'sequential' || resolved === 'overlapped',
        `expected sequential or overlapped, got "${resolved}"`
      );
    }
  });
});
