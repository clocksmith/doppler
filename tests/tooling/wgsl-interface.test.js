import assert from 'node:assert/strict';
import { compareUniformInterface, projectWgslInterface, reflectWgslInterface } from '../../tools/lib/wgsl-interface.js';

const source = `
/* fn fake() { /* nested } */ } */
struct Params { enabled: u32, padding: vec3<u32>, value: f32, }
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
fn helper() -> f32 { if true { return 1.0; } return 0.0; }
@compute @workgroup_size(1) fn main() { output[0] = params.value + helper(); }
`;
const projected = projectWgslInterface(source);
assert.ok(!projected.includes('return 1.0'));
assert.ok(projected.includes('fn helper() -> f32 {}'));
assert.ok(projected.includes('struct Params'));
const reflected = reflectWgslInterface(source);
assert.deepEqual(reflected.uniforms[0].fields.map(({ name, offset }) => [name, offset]), [
  ['enabled', 0], ['padding', 16], ['value', 28],
]);
assert.equal(reflected.uniforms[0].size, 32);
assert.deepEqual(reflected.entryPoints, ['main']);
assert.deepEqual(reflected.bindings.map(({ index, type }) => [index, type]), [[2, 'uniform'], [5, 'storage']]);
const uniforms = { size: 32, fields: reflected.uniforms[0].fields };
assert.deepEqual(compareUniformInterface(uniforms, reflected.uniforms, 'example'), []);
assert.match(compareUniformInterface({ ...uniforms, size: 16 }, reflected.uniforms, 'example').join(), /size 16/);
const wrong = structuredClone(uniforms);
wrong.fields[1].offset = 4;
assert.match(compareUniformInterface(wrong, reflected.uniforms, 'example').join(), /byte 16/);
assert.throws(() => projectWgslInterface('fn broken() { if true { }'), /unbalanced/);
console.log('wgsl-interface.test: passed');
