import assert from 'node:assert/strict';
import { createZeroDeltaPeftFixture, prepareAdapterEvaluationSource } from '../../tools/prepare-adapter-evaluation-source.js';
import { loadLoRAFromManifest } from '../../src/experimental/adapters/lora-loader.js';
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

const bytes = createZeroDeltaPeftFixture(2560, 4096);
assert.deepEqual(bytes, createZeroDeltaPeftFixture(2560, 4096));
const manifest = { id: 'fixture', baseModel: 'exact-base', rank: 1, alpha: 2, targetModules: ['q_proj'],
  weightsFormat: 'safetensors', weightsPath: 'fixture.safetensors', weightsSize: bytes.length,
  checksumAlgorithm: 'sha256', checksum: 'sha256:' + createHash('sha256').update(bytes).digest('hex') };
const adapter = await loadLoRAFromManifest(manifest, { readFile: async () => bytes, weightsLayout: 'peft' });
const pair = adapter.layers.get(0).q_proj;
assert.deepEqual(pair.aShape, [1, 2560]);
assert.deepEqual(pair.bShape, [4096, 1]);
assert(pair.a.every(value => Math.abs(value) === 1 / 256));
assert(pair.b.every(value => value === 0));
assert.equal(pair.scale, 2);
for (const value of [0, -1, NaN, 1.5]) assert.throws(() => createZeroDeltaPeftFixture(value, 4096));
const root = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-adapter-preparation-'));
const digest = data => 'sha256:' + createHash('sha256').update(data).digest('hex');
try {
  const original = JSON.stringify({ modelId: 'fixture-base', architecture: { hiddenSize: 2 },
    artifactIdentity: { manifestVariantId: 'original' }, tensors: { q: { shape: [3, 2] } },
    inference: { execution: { kernels: {}, mechanismKernels: [] } } });
  await fs.writeFile(path.join(root, 'manifest.json'), original);
  await fs.writeFile(path.join(root, 'weights.bin'), bytes);
  await fs.writeFile(path.join(root, 'kernel.wgsl'), 'test source bytes');
  const parent = JSON.stringify({ modelId: 'fixture-base', program: { manifestArtifactId: 'manifest' },
    wgslModules: [{ id: 'residual' }], artifacts: [
      { artifactId: 'manifest', path: 'manifest.json', hash: digest(original), sizeBytes: original.length },
      { artifactId: 'weights', path: 'weights.bin', hash: digest(bytes), sizeBytes: bytes.length },
    ] });
  await fs.writeFile(path.join(root, 'capsule.json'), parent);
  const config = { parentCapsule: path.join(root, 'capsule.json'), parentSha256: digest(parent),
    outputDir: path.join(root, 'candidate'), kernelRoot: root, projectionTensor: 'q', inputSize: 2, outputSize: 3,
    kernels: [{ id: 'adapter', file: 'kernel.wgsl', entry: 'main', sourceHash: digest('test source bytes'),
      digest: digest('test source bytes\n@@entry:main') }],
    residualModule: 'residual', manifestVariantId: 'separate', adapterId: 'fixture' };
  await prepareAdapterEvaluationSource(config);
  assert.equal(await fs.readFile(path.join(root, 'manifest.json'), 'utf8'), original);
  assert.equal(await fs.readFile(path.join(root, 'capsule.json'), 'utf8'), parent);
  const candidate = JSON.parse(await fs.readFile(path.join(config.outputDir, 'manifest.json'), 'utf8'));
  assert.equal(candidate.artifactIdentity.manifestVariantId, 'separate');
  assert.deepEqual(candidate.inference.execution.mechanismKernels, ['adapter']);
  assert.equal((await fs.stat(path.join(root, 'weights.bin'))).ino,
    (await fs.stat(path.join(config.outputDir, 'weights.bin'))).ino);
  await assert.rejects(prepareAdapterEvaluationSource(config), /EEXIST/);
  await assert.rejects(prepareAdapterEvaluationSource({ ...config, parentSha256: 'changed' }), /Retained Capsule changed/);
  await assert.rejects(prepareAdapterEvaluationSource({ ...config, outputDir: path.join(root, 'bad-digest'),
    kernels: [{ ...config.kernels[0], digest: config.kernels[0].sourceHash }],
  }), /kernel digest mismatch/);
} finally { await fs.rm(root, { recursive: true, force: true }); }
console.log('adapter-evaluation-fixture: ok');
