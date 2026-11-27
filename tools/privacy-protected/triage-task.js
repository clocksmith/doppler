// Frozen evaluation of two-word support tickets. The model emits a queue and an urgency label.
import { createHash } from 'node:crypto';

const issues = [
  ['network', 'wifi', 'ethernet', 'router', 'signal'],
  ['accounts', 'login', 'password', 'account', 'access'],
  ['hardware', 'printer', 'camera', 'laptop', 'monitor'],
  ['billing', 'invoice', 'payment', 'charge', 'refund'],
];
const urgency = [
  ['urgent', 'down', 'broken', 'blocked', 'urgent'],
  ['routine', 'slow', 'late', 'question', 'minor'],
];
export const TASK_ID = 'doppler.support-ticket-triage/v1';
export const TASK_DIGEST = 'sha256:ad127b8abd2f5b5074acb555786f3b2f3626960c23e370113fb91881d75b42da';
export const WORDS = Object.freeze([
  ...issues.flatMap(row => row.slice(1)), ...urgency.flatMap(row => row.slice(1)),
  ...issues.map(row => row[0]), 'routine', 'unknown', 'other', 'none',
]);
export const CONTEXT = 3;
export const GENERATED = 2;
export const QUALITY_FLOOR = 0.95;
export const MAX_QUALITY_DROP = 0.02;
export const MAX_CKKS_LOGIT_ERROR = 0.05;
export const WARM_BYTE_TARGET = 5_000_000;

export function taskCases(split) {
  if (!['train', 'validation', 'holdout'].includes(split)) throw new Error('Unknown task split');
  const rows = [];
  for (const [i, [queue, ...nouns]] of issues.entries()) {
    for (const [j, noun] of nouns.entries()) {
      for (const [k, [priority, ...descriptors]] of urgency.entries()) {
        for (const [l, descriptor] of descriptors.entries()) {
          const fold = (i * 4 + j + k * 4 + l) % 4;
          const partition = ['holdout', 'validation', 'train', 'train'][fold];
          if (partition !== split) continue;
          for (const prompt of [[noun, descriptor], [descriptor, noun]]) {
            rows.push({ prompt, tokens: prompt.map(word => WORDS.indexOf(word)),
              expected: [queue, priority].map(word => WORDS.indexOf(word)) });
          }
        }
      }
    }
  }
  return rows;
}

export function verifyTaskDigest() {
  const task = { id: TASK_ID, words: WORDS, context: CONTEXT, outputs: GENERATED,
    gates: [QUALITY_FLOOR, MAX_QUALITY_DROP, MAX_CKKS_LOGIT_ERROR, WARM_BYTE_TARGET],
    splits: ['train', 'validation', 'holdout'].map(taskCases) };
  const digest = 'sha256:' + createHash('sha256').update(JSON.stringify(task)).digest('hex');
  if (digest !== TASK_DIGEST) throw new Error('The frozen ticket task differs from its digest');
  return digest;
}

// Score each field against the ticket labels, with a separate full-answer score.
export function scoreAnswers(rows) {
  const count = rows.length;
  const queue = rows.filter(({ output, expected }) => output[0] === expected[0]).length;
  const urgency = rows.filter(({ output, expected }) => output[1] === expected[1]).length;
  const complete = rows.filter(({ output, expected }) => output[0] === expected[0] && output[1] === expected[1]).length;
  return { count, queue, urgency, complete, score: count ? complete / count : null };
}
