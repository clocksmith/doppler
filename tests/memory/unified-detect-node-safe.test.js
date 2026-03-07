import assert from 'node:assert/strict';

const { detectUnifiedMemory } = await import('../../src/memory/unified-detect.js');

const originalNavigatorDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'navigator');

try {
  if (originalNavigatorDescriptor) {
    delete globalThis.navigator;
  }

  const result = await detectUnifiedMemory();
  assert.equal(result.isUnified, false);
  assert.equal(result.reason, 'WebGPU not available');
} finally {
  if (originalNavigatorDescriptor) {
    Object.defineProperty(globalThis, 'navigator', originalNavigatorDescriptor);
  }
}

console.log('unified-detect-node-safe.test: ok');
