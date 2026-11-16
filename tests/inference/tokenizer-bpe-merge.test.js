import assert from 'node:assert/strict';
import { mergeBpeTokens } from '../../src/inference/tokenizers/bundled/merge.js';

// Independent reference retains the previous scan/copy implementation so rank
// order, simultaneous occurrences and overlapping pairs stay observable.
function reference(text, ranks) {
  let tokens = text.split('');
  while (tokens.length > 1) {
    let selected = null;
    let best = Infinity;
    for (let index = 0; index + 1 < tokens.length; index++) {
      const pair = `${tokens[index]} ${tokens[index + 1]}`;
      const rank = ranks.get(pair);
      if (rank !== undefined && rank < best) {
        selected = pair;
        best = rank;
      }
    }
    if (selected === null) break;
    const [left, right] = selected.split(' ');
    const merged = [];
    for (let index = 0; index < tokens.length; index++) {
      if (tokens[index] === left && tokens[index + 1] === right) {
        merged.push(left + right);
        index++;
      } else merged.push(tokens[index]);
    }
    tokens = merged;
  }
  return tokens;
}

for (const [text, rules] of [
  ['', []], ['a', []], ['abc', []],
  ['aaaaaaa', ['a a', 'aa a', 'aa aa']],
  // The new lower-ranked pair must wait until both original a+b pairs merge.
  ['abab', ['ab a', 'a b']],
  ['abcabc', ['a b', 'ab c', 'c ab', 'abc abc']],
  ['abcbcabcb', ['b c', 'a bc', 'bc ab', 'abc bc']],
  ['éé🙂🙂éé', ['é é', '\ud83d \ude42', '🙂 🙂']],
]) {
  const ranks = new Map(rules.map((rule, rank) => [rule, rank]));
  assert.deepEqual(mergeBpeTokens(text, ranks), reference(text, ranks), text);
}

let seed = 12345;
function random(bound) {
  seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
  return seed % bound;
}
const alphabet = ['a', 'b', 'c'];
const pieces = [...alphabet];
for (const left of alphabet) for (const right of alphabet) pieces.push(left + right);
for (let trial = 0; trial < 1000; trial++) {
  const rules = new Set();
  for (let rule = 0; rule < 40; rule++) rules.add(`${pieces[random(pieces.length)]} ${pieces[random(pieces.length)]}`);
  const ranks = new Map([...rules].map((rule, rank) => [rule, rank]));
  const text = Array.from({ length: 1 + random(150) }, () => alphabet[random(alphabet.length)]).join('');
  assert.deepEqual(mergeBpeTokens(text, ranks), reference(text, ranks), `trial ${trial}: ${text}`);
}

// Count actual rank lookups, independent of noisy wall-clock thresholds. Each
// original pair has a different rank, forcing the scan-based implementation
// to revisit the shrinking input once per pair.
const characters = Array.from({ length: 2048 }, (_, index) => String.fromCharCode(0x100 + index));
const ranks = new Map();
for (let index = 0; index < characters.length; index += 2) {
  ranks.set(`${characters[index]} ${characters[index + 1]}`, index / 2);
}
let lookups = 0;
const countedRanks = { get(key) { lookups++; return ranks.get(key); } };
const text = characters.join('');
const expected = reference(text, countedRanks);
const referenceLookups = lookups;
lookups = 0;
assert.deepEqual(mergeBpeTokens(text, countedRanks), expected);
assert.ok(lookups < text.length * 4, `adjacency updates made ${lookups} rank lookups`);
assert.ok(referenceLookups > lookups * 32, `${referenceLookups} scan lookups versus ${lookups} adjacent lookups`);

console.log('tokenizer-bpe-merge.test: ok');
