import assert from 'node:assert/strict';
import { matchesStopSequence } from '../../src/inference/pipelines/text/stopping.js';
const tokenizer = { decode: ids => String.fromCodePoint(...ids) };
const ids = text => [...text].map(value => value.codePointAt(0));
assert.equal(matchesStopSequence(tokenizer, ids('promptSTOP'), 6, ['STOP']), true);
assert.equal(matchesStopSequence(tokenizer, ids('STOPanswer'), 4, ['STOP']), false);
assert.equal(matchesStopSequence(tokenizer, ids('xSTOPmore'), 0, ['STOP']), false);
assert.equal(matchesStopSequence(tokenizer, ids('x終'), 0, ['end', '終']), true);
assert.equal(matchesStopSequence({ decode() { throw new Error('Must not decode'); } }, [], 0, []), false);
