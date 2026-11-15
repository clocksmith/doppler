/** Public, eight-symbol bigram fixture. This does not add a supported Doppler model family. */
import { createHash } from 'node:crypto';
import policy from './policy.json' with { type: 'json' };
export const counts = policy.vocabulary.map(() => policy.vocabulary.map(() => 1));
for (let i = 1; i < policy.corpus.length; i++) {
  counts[policy.vocabulary.indexOf(policy.corpus[i])][policy.vocabulary.indexOf(policy.corpus[i - 1])]++;
}
for (const row of counts) Object.freeze(row);
Object.freeze(counts);
export const modelId = 'sha256:' + createHash('sha256').update(JSON.stringify({ vocabulary: policy.vocabulary, counts })).digest('hex');
export const transitionScores = token => counts.map(row => row[token]);
export const choose = scores => scores.indexOf(Math.max(...scores));
export function reference(seed) {
  let token = policy.vocabulary.indexOf(seed), output = '';
  if (token < 0) throw Error('Fixture seed is outside the vocabulary');
  for (let step = 0; step < policy.steps; step++) { token = choose(transitionScores(token)); output += policy.vocabulary[token]; }
  return output;
}
