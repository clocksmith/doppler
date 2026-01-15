import { describe, it, expect } from 'vitest';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

function softmaxCpu(logits, rows, cols) {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r += 1) {
    let max = -Infinity;
    for (let c = 0; c < cols; c += 1) {
      const v = logits[r * cols + c];
      if (v > max) max = v;
    }
    let sum = 0;
    for (let c = 0; c < cols; c += 1) {
      const exp = Math.exp(logits[r * cols + c] - max);
      out[r * cols + c] = exp;
      sum += exp;
    }
    const inv = sum > 0 ? 1 / sum : 0;
    for (let c = 0; c < cols; c += 1) {
      out[r * cols + c] *= inv;
    }
  }
  return out;
}

describe('training/e2e-parity', () => {
  it('matches fixture loss mean', async () => {
    const path = resolve('tests/fixtures/training/python-parity.json');
    const raw = await readFile(path, 'utf-8');
    const fixture = JSON.parse(raw);
    const logits = new Float32Array(fixture.logits);
    const targets = fixture.targets;
    const softmax = softmaxCpu(logits, fixture.rows, fixture.cols);
    let sum = 0;
    for (let r = 0; r < fixture.rows; r += 1) {
      const idx = r * fixture.cols + targets[r];
      sum += -Math.log(Math.max(softmax[idx], 1e-9));
    }
    const mean = sum / fixture.rows;
    expect(Math.abs(mean - fixture.lossMean)).toBeLessThan(fixture.tolerance);
  });
});
