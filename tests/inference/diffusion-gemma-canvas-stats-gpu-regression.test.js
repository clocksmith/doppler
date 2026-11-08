import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';

const gpuProbe = await probeNodeGPU();
if (!gpuProbe.ready) {
  console.log(`diffusion-gemma-canvas-stats-gpu-regression.test: skipped (${gpuProbe.reason})`);
  process.exit(0);
}
const { runDiffusionGemmaCanvasStats } = await import('../../src/gpu/kernels/diffusion-gemma-sampling.js');
const { acquireBuffer, uploadData, readBuffer, releaseBuffer } = await import('../../src/memory/buffer-pool.js');
const { destroyDevice } = await import('../../src/gpu/device.js');

// The kernel's argmax, entropy and inverse-CDF sample must match a float64
// reference computed with the same temperature, pad exclusion and draws.
function reference(logits, { canvasLength, vocabSize, temperature, padTokenId, draws, softcap }) {
  const out = { argmax: [], sampled: [], entropy: [] };
  for (let row = 0; row < canvasLength; row += 1) {
    const values = [];
    for (let t = 0; t < vocabSize; t += 1) {
      const raw = logits[row * vocabSize + t];
      const capped = softcap > 0 ? softcap * Math.tanh(raw / softcap) : raw;
      values.push(capped / temperature);
    }
    let max = -Infinity;
    values.forEach((v, t) => { if (t !== padTokenId && v > max) max = v; });
    let sum = 0; let weighted = 0; let argmax = -1; let best = -Infinity;
    const probs = values.map((v, t) => {
      if (t === padTokenId) return 0;
      if (v > best) { best = v; argmax = t; }
      const e = Math.exp(v - max); sum += e; weighted += e * v; return e;
    });
    const target = draws[row] * sum;
    let cumulative = 0; let sampled = -1;
    for (let t = 0; t < vocabSize; t += 1) {
      if (t === padTokenId) continue;
      cumulative += probs[t]; sampled = t;
      if (target <= cumulative) break;
    }
    out.argmax.push(argmax); out.sampled.push(sampled); out.entropy.push(Math.log(sum) + max - weighted / sum);
  }
  return out;
}

try {
  // Vocab larger than one workgroup so the block scan crosses thread blocks.
  const canvasLength = 5;
  const vocabSize = 1000;
  const padTokenId = 3;
  const temperature = 0.7;
  const softcap = 30;
  const logits = new Float32Array(canvasLength * vocabSize);
  for (let row = 0; row < canvasLength; row += 1) {
    for (let t = 0; t < vocabSize; t += 1) {
      logits[row * vocabSize + t] = Math.sin(row * 3.1 + t * 0.013) * 6 + (t % 97 === row ? 9 : 0);
    }
    logits[row * vocabSize + padTokenId] = 80; // pad must never win
  }
  const draws = Float32Array.from([0.01, 0.33, 0.5, 0.77, 0.999]);
  const expected = reference(logits, { canvasLength, vocabSize, temperature, padTokenId, draws, softcap });

  const logitsBuffer = acquireBuffer(logits.byteLength, undefined, 'test_logits');
  uploadData(logitsBuffer, logits);
  const stats = await runDiffusionGemmaCanvasStats(logitsBuffer, {
    canvasLength, vocabSize, temperature, padTokenId, logitSoftcap: softcap, uniformDraws: draws,
  });
  const argmax = Array.from(new Uint32Array(await readBuffer(stats.argmaxBuffer, canvasLength * 4)));
  const sampled = Array.from(new Uint32Array(await readBuffer(stats.sampledBuffer, canvasLength * 4)));
  const entropy = Array.from(new Float32Array(await readBuffer(stats.entropyBuffer, canvasLength * 4)));
  assert.deepEqual(argmax, expected.argmax, 'argmax matches the reference');
  assert.deepEqual(sampled, expected.sampled, 'inverse-CDF sample matches the reference for the same draws');
  for (let row = 0; row < canvasLength; row += 1) {
    assert.ok(Math.abs(entropy[row] - expected.entropy[row]) < 1e-3, `entropy row ${row}: ${entropy[row]} vs ${expected.entropy[row]}`);
    assert.notEqual(sampled[row], padTokenId);
  }
  assert.ok(new Set(sampled).size > 1, 'different draws reach different tokens');
  releaseBuffer(stats.argmaxBuffer); releaseBuffer(stats.entropyBuffer); releaseBuffer(stats.sampledBuffer);

  await assert.rejects(
    () => runDiffusionGemmaCanvasStats(logitsBuffer, { canvasLength, vocabSize, temperature, padTokenId: null, logitSoftcap: 0 }),
    /uniformDraws must be a Float32Array/
  );
  await assert.rejects(
    () => runDiffusionGemmaCanvasStats(logitsBuffer, { canvasLength, vocabSize, temperature, padTokenId: null, logitSoftcap: 0, uniformDraws: Float32Array.from([0, 0, 0, 0, 1]) }),
    /must lie in \[0, 1\)/
  );
  releaseBuffer(logitsBuffer);
  console.log('diffusion-gemma-canvas-stats-gpu-regression.test: ok (GPU)');
} finally {
  await destroyDevice?.();
}
