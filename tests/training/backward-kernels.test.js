import { describe, it, expect } from 'vitest';
import { loadBackwardRegistry } from '../../src/config/backward-registry-loader.js';

describe('training/backward-kernels', () => {
  it('loads backward registry entries', () => {
    const registry = loadBackwardRegistry();
    expect(registry.ops).toHaveProperty('matmul');
    expect(registry.ops).toHaveProperty('cross_entropy');
    expect(registry.ops).toHaveProperty('attention');
  });
});
