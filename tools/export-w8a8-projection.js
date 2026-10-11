// Node-only fixture export. No GPU execution or model qualification.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { quantizeW8A8Reference, matmulW8A8Reference } from '../tests/kernels/reference/w8a8.js';

const ROOT = fileURLToPath(new URL('../', import.meta.url));
const FIXTURE = 'tests/fixtures/w8a8-qualification.json';
const PHYSICAL = 'artifacts/w8a8-kernels-20261011/frozen-projections-physical.json';
const REFERENCE = 'tests/kernels/reference/w8a8.js';
const OPERANDS = 'tests/kernels/w8a8-physical.js';
const SHADERS = ['quantize_w8a8.wgsl', 'matmul_w8a8.wgsl', 'matmul_w8a8_packed.wgsl', 'rescale_w8a8.wgsl'];
const hash = bytes => createHash('sha256').update(bytes).digest('hex');

/** @param {unknown} config */
export function validateProjectionExportConfig(config) {
  assert(config && typeof config === 'object' && !Array.isArray(config), 'Explicit export config required');
  assert.deepEqual(Object.keys(config).sort(), ['outputDir', 'packageBundlePath'], 'Unknown or missing export field');
  for (const value of Object.values(config)) assert(typeof value === 'string' && value.length > 0, 'Export paths must be nonempty strings');
  return config;
}

/** Encode words explicitly little endian, independently of host byte order. */
function words(values, type) {
  const bytes = Buffer.alloc(values.length * 4);
  values.forEach((value, index) => {
    if (type === 'f32') bytes.writeFloatLE(value, index * 4);
    else if (type === 'i32') bytes.writeInt32LE(value, index * 4);
    else bytes.writeUInt32LE(value, index * 4);
  });
  return bytes;
}

/** @param {{packageBundlePath: string, outputDir: string}} config */
export async function exportW8A8Projection(config) {
  validateProjectionExportConfig(config);
  const fixtureBytes = await readFile(path.join(ROOT, FIXTURE));
  const physicalBytes = await readFile(path.join(ROOT, PHYSICAL));
  const referenceBytes = await readFile(path.join(ROOT, REFERENCE));
  const operandSourceBytes = await readFile(path.join(ROOT, OPERANDS));
  const fixture = JSON.parse(fixtureBytes), physical = JSON.parse(physicalBytes);
  assert.equal(hash(fixtureBytes), physical.fixture.sha256, 'Frozen fixture identity changed');
  assert.equal(hash(referenceBytes), physical.files[`/${REFERENCE}`], 'Independent oracle changed since physical verification');
  assert.equal(hash(operandSourceBytes), physical.files[`/${OPERANDS}`], 'Operand generator changed since physical verification');
  const { operators } = fixture;
  assert.equal(operators.seed, 7331, 'Frozen operand seed changed');
  assert(operators.M.includes(1) && operators.projections.some(row => row.name === 'query' && row.N === 1024 && row.K === 640), 'Verified query shape missing');
  assert.equal(physical.result.qualified, true, 'Physical qualification failed');
  for (const variant of ['w8a8_portable', 'w8a8_packed']) {
    const row = physical.result.rows.find(row => row.M === 1 && row.N === 1024 && row.K === 640 && row.variant === variant);
    assert(row?.packedBytesExact && row.accumulatorsExact && row.outputBitsExact, 'Projection lacks exact physical verification');
  }
  const bundle = path.resolve(config.packageBundlePath);
  const receiptBytes = await readFile(path.join(bundle, 'receipt.json'));
  const receipt = JSON.parse(receiptBytes);
  assert.equal(receipt.passed, true, 'Installed package smoke failed');
  assert.equal(path.basename(receipt.package.filename), receipt.package.filename, 'Package filename must be local');
  assert.equal(hash(await readFile(path.join(bundle, receipt.package.filename))), receipt.package.sha256, 'Package archive identity changed');
  const payloads = new Map();
  for (const name of SHADERS) {
    const source = `src/gpu/kernels/${name}`;
    const bytes = await readFile(path.join(bundle, 'consumer/node_modules/doppler-gpu', source));
    assert.equal(hash(bytes), physical.files[`/${source}`], `Packaged shader differs from verified shader: ${name}`);
    payloads.set(name, bytes);
  }
  // Never replace retained evidence. A later export failure leaves its directory visible.
  await mkdir(config.outputDir, { recursive: false });
  // Exactly the synthetic operand construction exercised by w8a8-physical.js.
  const values = (size, salt) => Float32Array.from({ length: size }, (_, i) => ((i * 37 + salt) % 255 - 127) / 8);
  const A = values(640, operators.seed), W = values(1024 * 640, operators.seed + 12);
  const qa = quantizeW8A8Reference(A, 1, 640), qw = quantizeW8A8Reference(W, 1024, 640);
  const expected = matmulW8A8Reference(qa.bytes, qw.bytes, 1, 1024, 640);
  payloads.set('activation.f32le.bin', words(A, 'f32'));
  payloads.set('weights.f32le.bin', words(W, 'f32'));
  payloads.set('activation.w8a8.bin', qa.bytes);
  payloads.set('weights.w8a8.bin', qw.bytes);
  payloads.set('expected.i32le.bin', words(expected.accumulators, 'i32'));
  payloads.set('expected-activation-scaled.f32le.bin', words(expected.activationScaled, 'f32'));
  payloads.set('expected.f32le.bin', words(expected.output, 'f32'));
  payloads.set('uniforms.u32le.bin', words([1, 1024, 640, 161], 'u32'));
  payloads.set('fixture.json', fixtureBytes);
  payloads.set('physical-verification.json', physicalBytes);
  payloads.set('independent-reference.js', referenceBytes);
  payloads.set('physical-operand-source.js', operandSourceBytes);
  payloads.set('package-receipt.json', receiptBytes);
  const binding = (binding, buffer, type) => ({ binding, buffer, type, offset: 0 });
  const uniform = binding(0, 'uniforms', 'uniform');
  const storage = (slot, buffer, writable = false) => binding(slot, buffer, writable ? 'storage' : 'read-only-storage');
  const stage = (id, shader, constants, dispatch, bindings) => ({ id, shader, entryPoint: 'main', constants, dispatch, group: 0, bindings });
  const dotBindings = [uniform, storage(1, 'packedActivation'), storage(2, 'packedWeights'), storage(3, 'accumulator', true), storage(4, 'status', true)];
  const manifest = {
    schema: 'doppler.w8a8-projection-fixture/v1',
    scope: 'Synthetic operands for a real Gemma query projection shape; independently verified W8A8 operation, not actual model weights or model acceptance',
    comparisonClass: 'fixed-operation', shape: { M: 1, N: 1024, K: 640 }, seed: operators.seed,
    layout: { byteOrder: 'little-endian', packing: 'w8a8-row-v1', paddedK: 640, rowStrideBytes: 644, scaleOffsetBytes: 0, dataOffsetBytes: 4 },
    package: receipt.package,
    provenance: { command: ['node', 'tools/export-w8a8-projection.js', '<config.json>'], invocation: process.argv, config,
      sourceCommit: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: ROOT, encoding: 'utf8' }).trim(),
      sourceStatus: execFileSync('git', ['status', '--short'], { cwd: ROOT, encoding: 'utf8' }),
      exporterSha256: hash(await readFile(fileURLToPath(import.meta.url))), fixturePath: FIXTURE, physicalPath: PHYSICAL,
      referencePath: REFERENCE, operandSourcePath: OPERANDS, originalPhysicalCommit: physical.commit },
    buffers: {
      uniforms: { size: 16, initial: 'uniforms.u32le.bin', usage: ['UNIFORM', 'COPY_DST'] },
      activation: { size: 2560, initial: 'activation.f32le.bin', usage: ['STORAGE', 'COPY_DST'] },
      packedWeights: { size: 659456, initial: 'weights.w8a8.bin', usage: ['STORAGE', 'COPY_DST'] },
      packedActivation: { size: 644, initial: 'zero', usage: ['STORAGE', 'COPY_SRC', 'COPY_DST'], expected: 'activation.w8a8.bin' },
      accumulator: { size: 4096, initial: 'zero', usage: ['STORAGE', 'COPY_SRC'], expected: 'expected.i32le.bin' },
      activationScaled: { size: 4096, initial: 'zero', usage: ['STORAGE', 'COPY_SRC'], expected: 'expected-activation-scaled.f32le.bin' },
      output: { size: 4096, initial: 'zero', usage: ['STORAGE', 'COPY_SRC'], expected: 'expected.f32le.bin' },
      status: { size: 4, initial: 'zero', usage: ['STORAGE', 'COPY_SRC', 'COPY_DST'], expectedU32: 0 },
    },
    stages: [
      stage('quantize', SHADERS[0], { WORKGROUP_SIZE: 256 }, [1, 1, 1], [uniform, storage(1, 'activation'), storage(2, 'packedActivation', true), storage(3, 'status', true)]),
      stage('portable-dot', SHADERS[1], { WORKGROUP_SIZE: 64 }, [16, 1, 1], dotBindings),
      stage('packed-dot', SHADERS[2], { WORKGROUP_SIZE: 64 }, [16, 1, 1], dotBindings),
      stage('activation-rescale', SHADERS[3], { WORKGROUP_SIZE: 256, INPUT_IS_I32: true, SCALE_BY_ROW: true }, [4, 1, 1], [uniform, storage(1, 'accumulator'), storage(2, 'packedActivation'), storage(3, 'activationScaled', true), storage(4, 'status', true)]),
      stage('weight-rescale', SHADERS[3], { WORKGROUP_SIZE: 256, INPUT_IS_I32: false, SCALE_BY_ROW: false }, [4, 1, 1], [uniform, storage(1, 'activationScaled'), storage(2, 'packedWeights'), storage(3, 'output', true), storage(4, 'status', true)]),
    ],
    sequences: { portable: ['quantize', 'portable-dot', 'activation-rescale', 'weight-rescale'], packed: ['quantize', 'packed-dot', 'activation-rescale', 'weight-rescale'] },
    acceptance: { exactAllOutputBytes: true, statusMustBeZero: true, packedRequires: ['packed_4x8_integer_dot_product'],
      completion: 'Submit all declared stages; await completion, map exact readbacks including status; reject errors before exposing output; unmap and destroy owned resources',
      timing: 'None in this export; compiler/provider/timing require separately frozen native comparison',
      projectionScope: 'F32 activation through packing, integer dot and two rescale passes; prepacked dot alone is a separate component scope' },
    claims: { newPhysicalExecution: false, nativeDoeAcceptance: false, modelAcceptance: false, performance: false },
    files: Object.fromEntries([...payloads].map(([name, bytes]) => [name, { sha256: hash(bytes), sizeBytes: bytes.length }])),
  };
  for (const [name, bytes] of payloads) await writeFile(path.join(config.outputDir, name), bytes);
  await writeFile(path.join(config.outputDir, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  return manifest;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  assert.equal(process.argv.length, 3, 'Usage: node tools/export-w8a8-projection.js <config.json>');
  await exportW8A8Projection(JSON.parse(await readFile(process.argv[2], 'utf8')));
}
