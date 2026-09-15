import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { generateKernelUniforms } from '../../tools/generate-kernel-uniforms.js';

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-uniform-migration-'));
try {
  await fs.mkdir(path.join(root, 'src/config/kernels'), { recursive: true });
  await fs.mkdir(path.join(root, 'src/gpu/kernels'), { recursive: true });
  await fs.writeFile(path.join(root, 'package.json'), '{"type":"module"}');
  await fs.copyFile(new URL('../../src/gpu/kernels/uniform-encoding.js', import.meta.url), path.join(root, 'src/gpu/kernels/uniform-encoding.js'));
  const registry = { operations: { probe: { baseBindings: [{ index: 0, name: 'uniforms', type: 'uniform' }],
    baseUniforms: { size: 16, fields: [
      { name: 'count', type: 'u32', offset: 0 }, { name: 'gain', type: 'f32', offset: 4 },
      { name: '_pad0', type: 'u32', offset: 8, padding: true }, { name: '_pad1', type: 'u32', offset: 12, padding: true },
    ] }, variants: { main: { wgsl: 'probe.wgsl', entryPoint: 'main', workgroup: [1, 1, 1] } },
  } } };
  const uniform = registry.operations.probe.baseUniforms;
  const writeRegistry = () => fs.writeFile(path.join(root, 'src/config/kernels/registry.json'), JSON.stringify(registry));
  const shader = (fields, body = '') => `struct Params { ${fields} }\n@group(0) @binding(0) var<uniform> u: Params;\n@compute @workgroup_size(1) fn main() { ${body} }`;
  const writeShader = (source) => fs.writeFile(path.join(root, 'src/gpu/kernels/probe.wgsl'), source);
  await writeRegistry();
  await writeShader(shader('count: u32, gain: f32, _pad0: u32, _pad1: u32,'));
  const first = await generateKernelUniforms(root);
  assert.equal(first.layouts, 1);
  // Moving a field requires agreement at the WGSL boundary, not a blind refresh.
  uniform.fields[0].offset = 4;
  uniform.fields[1] = { name: 'scale', type: 'f32', offset: 0 };
  await writeRegistry();
  await assert.rejects(generateKernelUniforms(root), /count must be u32 at byte 0/);
  uniform.fields.sort((a, b) => a.offset - b.offset);
  await writeRegistry();
  await writeShader(shader('scale: f32, count: u32, _pad0: u32, _pad1: u32,'));
  const changed = await generateKernelUniforms(root);
  assert.notDeepEqual([...changed.files], [...first.files]);
  assert.deepEqual([...await generateKernelUniforms(root).then((result) => result.files)], [...changed.files], 'generation is deterministic');
  for (const [file, content] of changed.files) {
    await fs.mkdir(path.dirname(path.join(root, file)), { recursive: true });
    await fs.writeFile(path.join(root, file), content);
  }
  const { UNIFORM_WRITERS } = await import(pathToFileURL(path.join(root, 'src/gpu/kernels/generated/uniform-writers.js')).href);
  const writer = UNIFORM_WRITERS['probe/main'];
  const bytes = new Uint8Array(writer.size).fill(0xFF);
  const view = new DataView(bytes.buffer);
  assert.throws(() => writer.write(view, { count: 19, gain: 1.25 }, 'probe/main'), /scale.*requires/);
  for (const values of [{ count: 0xFE123456, scale: -1.25 }, { count: 17, scale: 2.75 }]) {
    writer.write(view, values, 'probe/main');
    assert.equal(view.getFloat32(0, true), values.scale);
    assert.equal(view.getUint32(4, true), values.count);
    assert.deepEqual([...bytes.subarray(8)], Array(8).fill(0), 'padding overwrites reused bytes');
  }
  await writeShader(shader('scale: f32, count: u32, _pad0: u32, _pad1: u32,', 'let stride = u._pad0;'));
  await assert.rejects(generateKernelUniforms(root), /_pad0 is read by WGSL/);
  console.log('kernel-uniform-generation.test: changed layout, incompatible wrapper, reuse, deterministic generation, padding authority passed');
} finally { await fs.rm(root, { recursive: true, force: true }); }
