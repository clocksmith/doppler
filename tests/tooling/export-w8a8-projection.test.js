// Synthetic package control for the offline exporter; no GPU or package qualification.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, mkdir, readFile, writeFile, copyFile, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { exportW8A8Projection, validateProjectionExportConfig } from '../../tools/export-w8a8-projection.js';

const root = fileURLToPath(new URL('../../', import.meta.url));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
for (const invalid of [null, undefined, [], {}, { packageBundlePath: 'a' },
  { packageBundlePath: '', outputDir: 'b' }, { packageBundlePath: 'a', outputDir: null },
  { packageBundlePath: 'a', outputDir: 'b', M: 16 }]) {
  assert.throws(() => validateProjectionExportConfig(invalid));
}
const temporary = await mkdtemp(path.join(os.tmpdir(), 'doppler-w8-export-'));
try {
  const bundle = path.join(temporary, 'bundle');
  const kernelDir = path.join(bundle, 'consumer/node_modules/doppler-gpu/src/gpu/kernels');
  await mkdir(kernelDir, { recursive: true });
  const archive = Buffer.from('synthetic test archive, not an installable package');
  await writeFile(path.join(bundle, 'synthetic.tgz'), archive);
  await writeFile(path.join(bundle, 'receipt.json'), JSON.stringify({ passed: true,
    package: { filename: 'synthetic.tgz', sha256: hash(archive), version: 'synthetic-test-only' } }));
  for (const name of ['quantize_w8a8.wgsl', 'matmul_w8a8.wgsl', 'matmul_w8a8_packed.wgsl', 'rescale_w8a8.wgsl']) {
    await copyFile(path.join(root, 'src/gpu/kernels', name), path.join(kernelDir, name));
  }
  const outputDir = path.join(temporary, 'export');
  const config = { packageBundlePath: bundle, outputDir };
  const manifest = await exportW8A8Projection(config);
  assert.deepEqual(manifest.shape, { M: 1, N: 1024, K: 640 });
  assert.equal(manifest.claims.newPhysicalExecution, false);
  const physical = JSON.parse(await readFile(path.join(outputDir, 'physical-verification.json'), 'utf8'));
  assert.equal(hash(await readFile(path.join(outputDir, 'physical-operand-source.js'))),
    physical.files[`/${manifest.provenance.originalOperandSourcePath}`]);
  assert.deepEqual(manifest.sequences.packed, ['quantize', 'packed-dot', 'activation-rescale', 'weight-rescale']);
  for (const [name, identity] of Object.entries(manifest.files)) {
    const bytes = await readFile(path.join(outputDir, name));
    assert.equal(bytes.length, identity.sizeBytes);
    assert.equal(hash(bytes), identity.sha256);
  }
  for (const stage of manifest.stages) {
    assert(manifest.files[stage.shader]);
    assert.equal(stage.entryPoint, 'main');
    for (const binding of stage.bindings) assert(manifest.buffers[binding.buffer]);
  }
  assert.deepEqual(manifest.stages[2].dispatch, [16, 1, 1]);
  assert.deepEqual(manifest.stages[4].constants, { WORKGROUP_SIZE: 256, INPUT_IS_I32: false, SCALE_BY_ROW: false });
  const a = await readFile(path.join(outputDir, 'activation.w8a8.bin'));
  const w = await readFile(path.join(outputDir, 'weights.w8a8.bin'));
  const acc = await readFile(path.join(outputDir, 'expected.i32le.bin'));
  const intermediate = await readFile(path.join(outputDir, 'expected-activation-scaled.f32le.bin'));
  const final = await readFile(path.join(outputDir, 'expected.f32le.bin'));
  for (const n of [0, 1023]) {
    let sum = 0;
    for (let k = 0; k < 640; k++) sum += a.readInt8(4 + k) * w.readInt8(n * 644 + 4 + k);
    assert.equal(acc.readInt32LE(n * 4), sum);
    const scaled = Math.fround(Math.fround(sum) * a.readFloatLE(0));
    assert.equal(intermediate.readFloatLE(n * 4), scaled);
    assert.equal(final.readFloatLE(n * 4), Math.fround(scaled * w.readFloatLE(n * 644)));
  }
  const retained = await readFile(path.join(outputDir, 'manifest.json'));
  await assert.rejects(exportW8A8Projection(config), /EEXIST/);
  assert.deepEqual(await readFile(path.join(outputDir, 'manifest.json')), retained);
  await writeFile(path.join(kernelDir, 'matmul_w8a8.wgsl'), 'changed shader');
  await assert.rejects(exportW8A8Projection({ ...config, outputDir: path.join(temporary, 'bad-shader') }), /Packaged shader differs/);
  await writeFile(path.join(bundle, 'synthetic.tgz'), 'changed archive');
  await assert.rejects(exportW8A8Projection({ ...config, outputDir: path.join(temporary, 'bad-archive') }), /Package archive identity changed/);
} finally {
  await rm(temporary, { recursive: true, force: true });
}
console.log('export-w8a8-projection.test: passed (offline synthetic package controls)');
