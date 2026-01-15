import { describe, it, expect } from 'vitest';
import { DynamicLossScaler } from '../../src/training/loss-scaling.js';

describe('training/optimizer', () => {
  it('updates loss scale on overflow', () => {
    const scaler = new DynamicLossScaler({
      enabled: true,
      initialScale: 8,
      minScale: 1,
      maxScale: 16,
      scaleFactor: 2,
      backoffFactor: 0.5,
      growthInterval: 2,
      overflowCheck: true,
    });
    scaler.update(true);
    expect(scaler.scale).toBe(4);
    scaler.update(false);
    scaler.update(false);
    expect(scaler.scale).toBe(8);
  });
});
