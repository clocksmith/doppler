import assert from 'node:assert/strict';
import { classifyWord } from '../../src/inference/pipelines/text/generator/whole-token-program.js';

const positions = [40, 41, 42, 43];
const tokens = [7, 1234, 99, 5];
assert.deepEqual(classifyWord([5, 5, 5, 5], positions, tokens), { kind: 'const', value: 5 });
assert.deepEqual(classifyWord([7, 1234, 99, 5], positions, tokens), { kind: 'token' });
assert.deepEqual(classifyWord([41, 42, 43, 44], positions, tokens), { kind: 'affine', base: 41, slope: 1, pos0: 40 });
assert.deepEqual(classifyWord([1024, 1536, 2048, 2560], positions, tokens), { kind: 'affine', base: 1024, slope: 512, pos0: 40 });
// A float field that changes with position is not affine in its u32 bits.
const w = new Uint32Array(new Float32Array([1 / 41, 1 / 42, 1 / 43, 1 / 44]).buffer);
assert.equal(classifyWord(Array.from(w), positions, tokens), null);
// A word that breaks the affine prediction in a later step is rejected.
assert.equal(classifyWord([41, 42, 43, 45], positions, tokens), null);
// Non-consecutive positions never classify as affine.
assert.equal(classifyWord([1, 2, 3, 4], [1, 3, 4, 5], tokens), null);
console.log('whole-token-program-classify.test: ok');
