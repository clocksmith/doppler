import assert from 'node:assert/strict';
import { affineLastPosition, classifyWord } from '../../src/inference/pipelines/text/generator/whole-token-template.js';

const positions = [40, 41, 42, 43];
assert.deepEqual(classifyWord([5, 5, 5, 5], positions), { kind: 'const', value: 5 });
assert.deepEqual(classifyWord([41, 42, 43, 44], positions), { kind: 'affine', base: 41, slope: 1, pos0: 40 });
assert.deepEqual(classifyWord([1024, 1536, 2048, 2560], positions), { kind: 'affine', base: 1024, slope: 512, pos0: 40 });
// The input token is declared by its owner, never inferred: four different token ids
// are neither constant nor affine.
assert.equal(classifyWord([7, 1234, 99, 5], positions), null);
// A float field that changes with position is not affine in its u32 bits.
const w = new Uint32Array(new Float32Array([1 / 41, 1 / 42, 1 / 43, 1 / 44]).buffer);
assert.equal(classifyWord(Array.from(w), positions), null);
// A word that breaks the affine prediction in a later step is rejected.
assert.equal(classifyWord([41, 42, 43, 45], positions), null);
// The slope comes from a consecutive first pair; a later, distant or earlier position
// (a kept program verified in another generation) must lie on the same line.
assert.equal(classifyWord([1, 2, 3, 4], [1, 3, 4, 5]), null);
assert.deepEqual(classifyWord([41, 42, 43, 44, 71], [40, 41, 42, 43, 70]), { kind: 'affine', base: 41, slope: 1, pos0: 40 });
assert.deepEqual(classifyWord([1024, 1536, 2048, 2560, 0], [40, 41, 42, 43, 38]), { kind: 'affine', base: 1024, slope: 512, pos0: 40 });
assert.equal(classifyWord([41, 42, 43, 44, 72], [40, 41, 42, 43, 70]), null);
assert.equal(classifyWord([10, 11, 12, 13, 1], [10, 11, 12, 13, 31]), null);

// Explicit GPU windows can advance by one to four tokens. Their scalar controls
// must follow exact integer lines, without inferred fractions or u32 wrapping.
assert.deepEqual(classifyWord([100,120,130,160], [10,12,13,16], true), { kind:'affine', base:100, slope:10, pos0:10 });
assert.equal(classifyWord([100,121,130,160], [10,12,13,16], true), null);
assert.equal(classifyWord([100,120,130,161], [10,12,13,16], true), null);
assert.equal(classifyWord([0xfffffffe,0,1,2], [10,12,13,14], true), null);
assert.equal(classifyWord([100,120,130,160], [10,10,13,16], true), null);

// Affine words and copy offsets bound the positions a program may run.
const kvOffset = { kind: 'affine', base: 40 * 1024, slope: 1024, pos0: 40 };
// A 64-row buffer of 1024-byte rows takes a 1024-byte copy at rows 0..63.
assert.equal(affineLastPosition(kvOffset, 0, 64 * 1024 - 1024), 63);
assert.equal(affineLastPosition({ kind: 'affine', base: 0xfffffffd, slope: 1, pos0: 10 }, 0, 0xffffffff), 12);
assert.equal(affineLastPosition({ kind: 'affine', base: 3, slope: -1, pos0: 10 }, 0, 0xffffffff), 13);
assert.equal(affineLastPosition({ kind: 'const', value: 9 }, 0, 0xffffffff), Infinity);
assert.equal(affineLastPosition({ kind: 'token' }, 0, 0xffffffff), Infinity);
console.log('whole-token-program-classify.test: ok');
