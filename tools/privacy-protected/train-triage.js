// Reproduce the research checkpoint from the training fold and select width on validation.
import { writeFileSync } from 'node:fs';
import { activations, ACTIVATION_BOUND, evaluatePlaintext, KEY_BOUND, MODEL_SCHEMA, QUADRATIC_TERM } from './triage-model.js';
import { CONTEXT, TASK_ID, taskCases, WORDS } from './triage-task.js';

function generator(seed) {
  return () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 2 ** 32;
  };
}

function train(width) {
  const random = generator(20261127 + width);
  const vector = (n, scale = 0) => Array.from({ length: n }, () => scale * (2 * random() - 1));
  const matrix = (rows, cols, scale) => Array.from({ length: rows }, () => vector(cols, scale));
  const n = WORDS.length;
  const model = { schema: MODEL_SCHEMA, task: TASK_ID, context: CONTEXT, width,
    keys: vector(n, 0.1), values: matrix(n, width, 1), hidden: { weight: matrix(width, width, 0.7), bias: vector(width) },
    head: { weight: matrix(n, width, 1.3), bias: vector(n) } };
  const rows = taskCases('train');
  const rate = 0.028;
  for (let epoch = 0; epoch < 450; epoch++) {
    for (let i = rows.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [rows[i], rows[j]] = [rows[j], rows[i]];
    }
    for (const { tokens, expected } of rows) {
      for (const [step, target] of expected.entries()) {
        const prefix = step ? [...tokens, expected[0]] : tokens;
        const approximate = random() < 0.65;
        const { query, weights, pooled, hidden, activated, logits } = activations(model, prefix, approximate);
        const shifted = logits.map(x => Math.exp(x - Math.max(...logits)));
        const sum = shifted.reduce((a, b) => a + b);
        const dlogits = shifted.map((x, i) => x / sum - Number(i === target));
        const dHidden = vector(width);
        const dPool = vector(width);
        model.head.weight.forEach((row, i) => {
          row.forEach((weight, j) => {
            dHidden[j] += dlogits[i] * weight;
            row[j] -= rate * (dlogits[i] * activated[j] + 0.00015 * weight);
          });
          model.head.bias[i] -= rate * dlogits[i];
        });
        model.hidden.weight.forEach((row, i) => {
          const x = hidden[i];
          const s = Math.sqrt(2 / Math.PI);
          const t = Math.tanh(s * (x + 0.044715 * x ** 3));
          const derivative = approximate ? 0.5 + 2 * QUADRATIC_TERM * x
            : (1 + t) / 2 + x * (1 - t * t) * s * (1 + 3 * 0.044715 * x * x) / 2;
          const rangePenalty = 4 * Math.sign(x) * Math.max(Math.abs(x) - ACTIVATION_BOUND * 0.8, 0);
          const gradient = dHidden[i] * derivative + rangePenalty;
          row.forEach((weight, j) => {
            dPool[j] += gradient * weight;
            row[j] -= rate * gradient * pooled[j];
          });
          model.hidden.bias[i] -= rate * gradient;
        });
        prefix.forEach((token, i) => {
          let dScore = 0;
          model.values[token].forEach((x, j) => {
            dScore += dPool[j] * (approximate ? x / prefix.length : weights[i] * (x - pooled[j]));
            model.values[token][j] -= rate * dPool[j] * weights[i];
          });
          model.keys[token] = Math.min(KEY_BOUND, Math.max(-KEY_BOUND, model.keys[token] - rate * query * dScore));
        });
      }
    }
  }
  return model;
}

for (const width of [2, 4, 8, 16]) {
  const model = train(width);
  const { normal, approximate, passed } = evaluatePlaintext(model, 'validation');
  console.error(JSON.stringify({ width, normal, approximate, passed }));
  if (passed) {
    writeFileSync(new URL('./triage-weights.json', import.meta.url), JSON.stringify(model, null, 2) + '\n');
    break;
  }
  if (width === 16) throw new Error('No candidate passed validation');
}
