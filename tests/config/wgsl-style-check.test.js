import assert from 'node:assert/strict';
import { checkWgslSource, RULE_IDS } from '../../tools/check-wgsl-style.js';

const check = (source, path = 'fixture.wgsl') => checkWgslSource(source, path);
assert.deepEqual(check(`enable f16; override TILE_SIZE: u32 = 8u;
@binding(0) @group(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> x: array<f16>;
fn apply_gain() { let value = u.gain; }
// fn badName() {} u._pad
/* nested /* override bad: u32; */ f16 */`), []);
const cases = [
  ['override tileSize: u32;', 'wgsl-override-name'],
  ['fn badName() {}', 'wgsl-function-name'],
  ['fn main() { let flag = u._pad0; }', 'wgsl-padding-access'],
  ['var<private> value: f16;', 'wgsl-f16-enable'],
  ['var<private> value: vec2h;', 'wgsl-f16-enable'],
  ['/* missing end', 'wgsl-comments'],
  ['@group(0) @binding(1) var<storage> a: f32; @binding(1u) @group(0u) var<storage> b: f32;', 'wgsl-binding-unique'],
];
assert.deepEqual([...new Set(cases.map(([, id]) => id))].sort(), [...RULE_IDS].sort());
assert.equal(check('// 😀\nfn badName() {}')[0].line, 2);
for (const [source, ruleId] of cases) {
  assert.deepEqual(check(source).map(item => item.ruleId), [ruleId], source);
}
assert.deepEqual(check('var<private> value: f16;', 'helper.wgslinc'), []);
assert.deepEqual(check('// /* ignored\nfn main() {}'), []);
assert.equal(check('// first line\nfn badName() {}')[0].line, 2);
assert.deepEqual(check('@group(0) @binding(1) var<storage> a: f32; @group(1) @binding(1) var<storage> b: f32;'), []);
console.log('wgsl-style-check: positive, negative, nested-comment, include and line-location fixtures pass');
