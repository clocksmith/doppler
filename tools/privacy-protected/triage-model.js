import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { CONTEXT, GENERATED, MAX_QUALITY_DROP, QUALITY_FLOOR, scoreAnswers, taskCases, TASK_ID, WORDS } from './triage-task.js';

export const MODEL_SCHEMA = 'doppler.protected-triage-model/v1';
export const ACTIVATION_BOUND = 1.5;
export const KEY_BOUND = 0.25;
export const QUADRATIC_TERM = 0.3989422804014327;
export const DEFAULT_TRIAGE_MODEL = new URL('./triage-weights.json', import.meta.url);
const sha256 = value => 'sha256:' + createHash('sha256').update(value).digest('hex');

const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
export const queryAt = length => length === 2 ? 1 : -1;
export const argmax = values => values.reduce((best, value, i) => value > values[best] ? i : best, 0);

export function loadTriageModel(filename = DEFAULT_TRIAGE_MODEL) {
  const bytes = readFileSync(filename);
  const model = JSON.parse(bytes);
  const { width, keys, values, hidden, head } = model;
  const vector = (x, n) => Array.isArray(x) && x.length === n && x.every(Number.isFinite);
  const matrix = (x, m, n) => Array.isArray(x) && x.length === m && x.every(row => vector(row, n));
  if (model.schema !== MODEL_SCHEMA || model.task !== TASK_ID || model.context !== CONTEXT
    || ![2, 4, 8, 16].includes(width) || !vector(keys, WORDS.length) || keys.some(key => Math.abs(key) > KEY_BOUND)
    || !matrix(values, WORDS.length, width) || !matrix(hidden?.weight, width, width) || !vector(hidden?.bias, width)
    || !matrix(head?.weight, WORDS.length, width) || !vector(head?.bias, WORDS.length)) {
    throw new Error('Malformed pinned ticket triage model');
  }
  const id = sha256(bytes);
  const circuit = sha256(JSON.stringify({ schema: MODEL_SCHEMA, model: id, words: WORDS, activationBound: ACTIVATION_BOUND,
    quadratic: QUADRATIC_TERM, attention: 'linear-positional-query', context: CONTEXT, output: GENERATED }));
  return { model, id, circuit };
}

export function activations(model, prefix, approximate) {
  const query = queryAt(prefix.length);
  const scores = prefix.map(token => query * model.keys[token]);
  const weights = approximate ? scores.map(score => (1 + score) / prefix.length) : scores.map(Math.exp);
  if (!approximate) {
    const sum = weights.reduce((a, b) => a + b);
    for (let i = 0; i < weights.length; i++) weights[i] /= sum;
  }
  const pooled = Array.from({ length: model.width }, (_, j) => prefix.reduce(
    (sum, token, i) => sum + weights[i] * model.values[token][j], 0));
  const hidden = model.hidden.weight.map((row, i) => dot(row, pooled) + model.hidden.bias[i]);
  // The normal model uses GELU with its standard tanh approximation. The HE circuit uses the quadratic Taylor term.
  const activated = hidden.map(x => approximate ? 0.5 * x + QUADRATIC_TERM * x * x
    : 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3))));
  const logits = model.head.weight.map((row, i) => dot(row, activated) + model.head.bias[i]);
  return { query, weights, pooled, hidden, activated, logits };
}

export function evaluatePlaintext(model, split) {
  const rows = taskCases(split).map(({ prompt, tokens, expected }) => {
    const run = (approximate, forced) => {
      const prefix = [...tokens];
      const output = [];
      const logits = [];
      const rangeViolations = [];
      let nonFiniteLogits = 0;
      for (let step = 0; step < GENERATED; step++) {
        const trace = activations(model, prefix, approximate);
        const outside = trace.hidden.filter(x => !Number.isFinite(x) || Math.abs(x) > ACTIVATION_BOUND).length;
        if (outside) rangeViolations.push({ step: step + 1, activation: 'gelu', count: outside });
        const token = argmax(trace.logits);
        output.push(token);
        logits.push(trace.logits);
        nonFiniteLogits += trace.logits.filter(x => !Number.isFinite(x)).length;
        prefix.push(forced ? expected[step] : token);
      }
      return { output, logits, rangeViolations, nonFiniteLogits };
    };
    const normal = run(false, false), approximate = run(true, false);
    const sharedNormal = run(false, true), sharedApproximate = run(true, true);
    return { prompt, tokens, expected, normal, approximate, sharedNormal, sharedApproximate };
  });
  const summarize = key => ({
    ...scoreAnswers(rows.map(row => ({ expected: row.expected, output: row[key].output }))),
    rangeViolationPrompts: rows.filter(row => row[key].rangeViolations.length).length,
    nonFiniteLogits: rows.reduce((sum, row) => sum + row[key].nonFiniteLogits, 0),
  });
  const normal = summarize('normal'), approximate = summarize('approximate');
  const sharedPrefix = { normal: summarize('sharedNormal'), approximate: summarize('sharedApproximate') };
  const sharedPrefixMaxLogitError = Math.max(...rows.flatMap(row => row.sharedNormal.logits.flatMap(
    (logits, step) => logits.map((x, i) => Math.abs(x - row.sharedApproximate.logits[step][i])))));
  const passed = normal.score >= QUALITY_FLOOR && approximate.score >= QUALITY_FLOOR
    && normal.score - approximate.score <= MAX_QUALITY_DROP
    && !approximate.rangeViolationPrompts && !sharedPrefix.approximate.rangeViolationPrompts
    && !approximate.nonFiniteLogits && !sharedPrefix.approximate.nonFiniteLogits;
  return { task: TASK_ID, split, passed, normal, approximate, sharedPrefix, sharedPrefixMaxLogitError, rows };
}
