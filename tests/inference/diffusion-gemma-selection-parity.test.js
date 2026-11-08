import assert from 'node:assert/strict';
import {
  applyEntropyBoundStep,
  applyEntropyBoundStatsStep,
  createSeededRandom,
  denoiseCanvas,
  denoiseCanvasWithStatsProvider,
  drawCanvasUniforms,
} from '../../src/inference/pipelines/diffusion-gemma/sampling.js';

// Both denoising paths must select the same tokens from the same random
// stream: the CPU path samples per position while it scans the logits, the
// stats path receives one uniform draw per position and samples on the GPU.
// The GPU path used to accept argmax tokens (and report them as "sampled").
const config = {
  canvasLength: 4,
  maxDenoisingSteps: 3,
  tMin: 0.5,
  tMax: 1.0,
  entropyBound: 1.2,
  confidenceThreshold: 0.001,
  stabilityThreshold: 99,
  padTokenId: 0,
  vocabSize: 6,
};

// Reference statistics with the kernel's semantics: pad excluded, inverse CDF
// over vocabulary order with the supplied draw.
function referenceStats(logits, temperature, draws) {
  const argmaxCanvas = new Int32Array(config.canvasLength);
  const sampledCanvas = new Int32Array(config.canvasLength);
  const entropies = new Float32Array(config.canvasLength);
  for (let i = 0; i < config.canvasLength; i += 1) {
    const row = Array.from(logits.subarray(i * config.vocabSize, (i + 1) * config.vocabSize), (v) => v / temperature);
    let max = -Infinity;
    row.forEach((v, t) => { if (t !== config.padTokenId && v > max) max = v; });
    let sum = 0;
    let weighted = 0;
    let argmax = -1;
    let argmaxValue = -Infinity;
    const probs = row.map((v, t) => {
      if (t === config.padTokenId) return 0;
      if (v > argmaxValue) { argmaxValue = v; argmax = t; }
      const e = Math.exp(v - max);
      sum += e;
      weighted += e * v;
      return e;
    });
    const target = draws[i] * sum;
    let cumulative = 0;
    let sampled = -1;
    for (let t = 0; t < row.length; t += 1) {
      if (t === config.padTokenId) continue;
      cumulative += probs[t];
      sampled = t;
      if (target <= cumulative) break;
    }
    argmaxCanvas[i] = argmax;
    sampledCanvas[i] = sampled;
    entropies[i] = Math.log(sum) + max - weighted / sum;
  }
  return { argmaxCanvas, sampledCanvas, entropies };
}

// Logits where pad (token 0) has the largest value everywhere and the rest are
// spread so sampling and argmax differ for some draws.
function logitsFor(step) {
  const logits = new Float32Array(config.canvasLength * config.vocabSize);
  for (let i = 0; i < config.canvasLength; i += 1) {
    for (let t = 0; t < config.vocabSize; t += 1) {
      logits[i * config.vocabSize + t] = t === 0 ? 50 : Math.sin(i * 1.7 + t * 0.9 + step) * 2.5;
    }
  }
  return logits;
}

{
  const logits = logitsFor(1);
  const draws = Float32Array.from([0.05, 0.5, 0.95, 0.3]);
  const reference = referenceStats(logits, 0.8, draws);
  assert.ok(reference.argmaxCanvas.every((token) => token !== 0), 'pad is never the argmax');
  assert.ok(reference.sampledCanvas.every((token) => token !== 0), 'pad is never sampled');

  // CPU path: the same draws come from the random stream in position order.
  let cursor = 0;
  const randomCpu = () => (cursor < draws.length ? draws[cursor++] : 0.42);
  const cpu = applyEntropyBoundStep(Int32Array.from([1, 1, 1, 1]), logits, config, { temperature: 0.8, random: randomCpu });
  assert.deepEqual([...cpu.argmaxCanvas], [...reference.argmaxCanvas]);
  assert.deepEqual([...cpu.sampledCanvas], [...reference.sampledCanvas]);
  for (let i = 0; i < config.canvasLength; i += 1) {
    assert.ok(Math.abs(cpu.entropies[i] - reference.entropies[i]) < 1e-5);
  }

  // Stats path: accepted positions take the sampled token, not the argmax.
  const stats = applyEntropyBoundStatsStep(Int32Array.from([1, 1, 1, 1]), reference, config, { temperature: 0.8, random: () => 0.42 });
  assert.deepEqual([...stats.sampledCanvas], [...reference.sampledCanvas]);
  assert.deepEqual([...stats.argmaxCanvas], [...reference.argmaxCanvas]);
  for (let i = 0; i < config.canvasLength; i += 1) {
    if (stats.accepted[i]) assert.equal(stats.canvas[i], reference.sampledCanvas[i]);
  }
  assert.ok(stats.acceptedCount > 0 && stats.acceptedCount < config.canvasLength, 'the fixture accepts some positions and rejects others');
  assert.deepEqual([...stats.canvas], [...cpu.canvas], 'both paths produce the same next canvas from the same draws');
  assert.throws(
    () => applyEntropyBoundStatsStep(Int32Array.from([1, 1, 1, 1]), { argmaxCanvas: reference.argmaxCanvas, entropies: reference.entropies }, config, { temperature: 0.8, random: () => 0.1 }),
    /sampledCanvas must be an array or typed array/
  );
}

// Whole denoising loops agree step for step when the stats provider samples
// with the draws it is handed.
{
  const seed = 2026;
  const cpuResult = await denoiseCanvas(config, {
    random: createSeededRandom(seed),
    logitsProvider: async ({ step }) => logitsFor(step),
  });
  const seen = [];
  const statsResult = await denoiseCanvasWithStatsProvider(config, {
    random: createSeededRandom(seed),
    statsProvider: async ({ step, temperature, uniformDraws }) => {
      seen.push(uniformDraws.length);
      return { ...referenceStats(logitsFor(step), temperature, uniformDraws), selfConditioningLogits: null };
    },
  });
  assert.deepEqual(seen, [4, 4, 4]);
  assert.equal(cpuResult.stepsRun, statsResult.stepsRun);
  assert.deepEqual([...cpuResult.canvas], [...statsResult.canvas]);
  assert.deepEqual([...cpuResult.argmaxCanvas], [...statsResult.argmaxCanvas]);
  assert.deepEqual([...cpuResult.lastStep.sampledCanvas], [...statsResult.lastStep.sampledCanvas]);
}

const draws = drawCanvasUniforms(createSeededRandom(7), 5);
assert.equal(draws.length, 5);
assert.ok(draws.every((d) => d >= 0 && d < 1));
console.log('diffusion gemma selection parity ok');
