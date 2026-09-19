import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { KERNEL_CONFIGS, getKernelConfig } from '../../src/config/kernel-registry-contract.js';
import { getUniformByteLength, writeUniformsFromObject, createKernelUniformBuffer } from '../../src/gpu/kernels/uniform-utils.js';
import { reflectWgslInterface } from '../../tools/lib/wgsl-interface.js';
import { writeI32 } from '../../src/gpu/kernels/uniform-encoding.js';

const sources = new Map();
let variants = 0;
let fields = 0;
for (const [op, configs] of Object.entries(KERNEL_CONFIGS)) {
  for (const [variant, config] of Object.entries(configs)) {
    if (config.uniforms === null) { assert.equal(getUniformByteLength(config), 0); continue; }
    if (!sources.has(config.shaderFile)) {
      const source = await fs.readFile(new URL(`../../src/gpu/kernels/${config.shaderFile}`, import.meta.url), 'utf8');
      sources.set(config.shaderFile, reflectWgslInterface(source).uniforms[0]);
    }
    const actual = sources.get(config.shaderFile);
    assert.equal(getUniformByteLength(config), actual.size, `${op}/${variant} size`);
    const values = {};
    for (const field of config.uniforms.fields) if (!field.padding) {
      values[field.name] = field.type === 'f32' ? -(field.offset + 0.125) : 0xF0000000 + field.offset;
    }
    const data = new Uint8Array(actual.size + 8).fill(0xAA);
    const view = new DataView(data.buffer, 4, actual.size);
    writeUniformsFromObject(view, config, values);
    assert.deepEqual([...data.subarray(0, 4)], [170, 170, 170, 170]);
    assert.deepEqual([...data.subarray(-4)], [170, 170, 170, 170]);
    for (const field of actual.fields) {
      const declared = config.uniforms.fields.find((candidate) => candidate.name === field.name);
      const got = field.type === 'f32' ? view.getFloat32(field.offset, true) : view.getUint32(field.offset, true);
      assert.equal(got, declared.padding ? 0 : values[field.name], `${op}/${variant}.${field.name}`);
      if (!declared.padding) {
        const omitted = { ...values };
        delete omitted[field.name];
        assert.throws(() => writeUniformsFromObject(view, config, omitted), /requires/, `${op}/${variant}.${field.name} is required`);
        for (const invalid of field.type === 'f32' ? [NaN, Infinity, 1e100, null] : [-1, 0x100000000, 1.5, null]) {
          assert.throws(() => writeUniformsFromObject(view, config, { ...values, [field.name]: invalid }), /requires/);
        }
      }
      fields++;
    }
    assert.throws(() => writeUniformsFromObject(new DataView(new ArrayBuffer(actual.size - 1)), config, values), /uniform view/);
    variants++;
  }
}
const sample = getKernelConfig('sample', 'argmax');
assert.ok(Object.isFrozen(sample.uniforms));
assert.ok(Object.isFrozen(sample.uniforms.fields[0]));
assert.throws(() => { sample.uniforms = { ...sample.uniforms, size: 16 }; }, /read only/);
assert.throws(() => writeUniformsFromObject(new DataView(new ArrayBuffer(32)), {
  ...sample, uniforms: { ...sample.uniforms, size: 16 },
}, {}), /immutable registry/);
let allocations = 0;
assert.throws(() => createKernelUniformBuffer('invalid', sample, {}, null, {
  createBuffer() { allocations++; throw new Error('must not allocate'); },
}), /requires/);
assert.equal(allocations, 0, 'missing required fields fail before GPU allocation');
const signed = new DataView(new ArrayBuffer(4));
for (const value of [-0x80000000, 0, 0x7FFFFFFF]) { writeI32(signed, 0, value, 'test', 'value'); assert.equal(signed.getInt32(0, true), value); }
for (const value of [-0x80000001, 0x80000000, undefined, 0.5]) assert.throws(() => writeI32(signed, 0, value, 'test', 'value'), /i32/);
console.log(`kernel-uniform-writers.test: ${variants} variants, ${fields} fields passed`);
