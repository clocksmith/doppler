import { describe, it, expect } from 'vitest';
import { AutogradTape, OpType } from '../../src/training/autograd.js';
import { loadBackwardRegistry } from '../../src/config/backward-registry-loader.js';

describe('training/autograd', () => {
  it('records ops and resets', async () => {
    const tape = new AutogradTape(loadBackwardRegistry());
    const value = await tape.record(OpType.SCALE, async (x) => x + 1, [1], { scale: 1 });
    expect(value).toBe(2);
    expect(tape.records.length).toBe(1);
    tape.reset();
    expect(tape.records.length).toBe(0);
  });
});
