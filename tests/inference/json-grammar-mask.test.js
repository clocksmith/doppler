import assert from 'node:assert/strict';
import { createJsonGrammarMask } from '../../src/inference/pipelines/structured/json-grammar-mask.js';

const chars = [...new Set(' {}[],:"\\/bfnrttruefalsenull0123456789-.eE+abcdefABCDEFxyz\t\r\n`')];
const pieces = ['', ...chars, '{"count":3}', '{"bad":}', '"x}y"', '\\u0022', '```json\n{', 'null,}', 'false]', ' ', '{"a":1,}'];
const eos = 0;
const tokenizer = { decode: ids => ids.map(id => pieces[id]).join(''), getSpecialTokens: () => ({ eos }) };
const mask = createJsonGrammarMask({ tokenizer });
const encode = text => [...text].map(char => { const id = pieces.indexOf(char); assert.ok(id > 0, char); return id; });
const allowed = prefix => {
  const logits = new Float32Array(pieces.length);
  mask(logits, { generatedIds: encode(prefix), tokenizer, vocabSize: logits.length });
  return id => Number.isFinite(logits[id]);
};
const token = text => pieces.indexOf(text);
for (const full of ['{}', '{"x":"}\\\"["}', '{"x":[true,false,null,-0.12e+3,{"y":2}]}', '{"x":"\\u0022"}']) {
  for (let end = 0; end < full.length; end += 1) {
    assert.equal(allowed(full.slice(0,end))(token(full[end])), true, `${full} at ${end}`);
    assert.equal(allowed(full.slice(0,end))(eos), false);
  }
  assert.equal(allowed(full)(eos), true);
  assert.equal(allowed(full)(token('{')), false);
}
for (const [prefix, next] of [['', '`'], ['', '['], ['{', 'x'], ['{"x"', ','], ['{"x":', '}'],
  ['{"x":1,', '}'], ['{"x":[1,', ']'], ['{"x":0', '1'], ['{"x":1.', '}'], ['{"x":1e', '}'],
  ['{"x":"', '\n'], ['{"x":"\\', 'x'], ['{"x":"\\u0', 'x'], ['{"x":true', 'x']]) {
  assert.equal(allowed(prefix)(token(next)), false, `${prefix} + ${next}`);
}
assert.equal(allowed('')(token('{"count":3}')), true);
for (const invalid of ['{"bad":}', '```json\n{', '{"a":1,}']) assert.equal(allowed('')(token(invalid)), false);
assert.equal(allowed('{"x":')(token('"x}y"')), true);
assert.equal(allowed('{"x":')(token('null,}')), false);
// Reuse and same-length revised prefixes must not retain an old parse state.
assert.equal(allowed('{}')(eos), true);
assert.equal(allowed('{"')(eos), false);
assert.equal(allowed('')(token('{')), true);
assert.throws(() => allowed('{bad'), /invalid generated prefix/);
assert.throws(() => createJsonGrammarMask()(new Float32Array(2), {generatedIds: []}), /tokenizer/);
assert.throws(() => createJsonGrammarMask({tokenizer:{decode(){throw new Error('decode failed');}}})(new Float32Array(2), {generatedIds: []}), /decode failed/);
console.log('json-grammar-mask.test: ok');
