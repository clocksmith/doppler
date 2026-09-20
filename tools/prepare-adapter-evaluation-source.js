#!/usr/bin/env node
// Preparation only: derive an unsigned source and a controlled PEFT fixture.
// Never edits the retained Capsule or claims trained-adapter quality.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { buildWgslClosure } from '../src/tooling/program-bundle/wgsl-closure.js';

const hash = bytes => `sha256:${createHash('sha256').update(bytes).digest('hex')}`;
const read = async file => JSON.parse(await fs.readFile(file, 'utf8'));
const write = async (file, data) => fs.writeFile(file, JSON.stringify(data, null, 2) + '\n', { flag: 'wx' });

export function createZeroDeltaPeftFixture(inputSize, outputSize) {
  assert(Number.isSafeInteger(inputSize) && inputSize > 0);
  assert(Number.isSafeInteger(outputSize) && outputSize > 0);
  // A is nonzero; B=0 gives an independent, exact zero-delta oracle. This
  // exercises both projections, scaling and addition but not nonzero quality.
  const header = Buffer.from(JSON.stringify({
    'base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight': {
      dtype: 'F32', shape: [1, inputSize], data_offsets: [0, inputSize * 4],
    },
    'base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight': {
      dtype: 'F32', shape: [outputSize, 1], data_offsets: [inputSize * 4, (inputSize + outputSize) * 4],
    },
  }));
  const headerSize = Math.ceil(header.length / 8) * 8;
  const bytes = Buffer.alloc(8 + headerSize + (inputSize + outputSize) * 4);
  bytes.writeBigUInt64LE(BigInt(headerSize));
  bytes.fill(32, 8, 8 + headerSize);
  header.copy(bytes, 8);
  for (let i = 0; i < inputSize; i++) bytes.writeFloatLE((i % 2 ? -1 : 1) / 256, 8 + headerSize + i * 4);
  return bytes;
}

export async function prepareAdapterEvaluationSource(config) {
  const parentBytes = await fs.readFile(config.parentCapsule);
  assert.equal(hash(parentBytes), config.parentSha256, 'Retained Capsule changed');
  const parent = JSON.parse(parentBytes);
  const root = path.dirname(config.parentCapsule);
  const output = path.resolve(config.outputDir);
  await fs.mkdir(output); // Fail rather than overwrite an earlier candidate.
  const source = { schema: 'doppler.adapter-evaluation-source/v1', modelId: parent.modelId,
    artifacts: structuredClone(parent.artifacts), wgslModules: structuredClone(parent.wgslModules),
    program: { manifestArtifactId: parent.program.manifestArtifactId } };
  const manifestArtifact = source.artifacts.find(row => row.artifactId === source.program.manifestArtifactId);
  const manifestBytes = await fs.readFile(path.join(root, manifestArtifact.path));
  assert.equal(hash(manifestBytes), manifestArtifact.hash);
  const manifest = JSON.parse(manifestBytes);
  const tensor = manifest.tensors[config.projectionTensor];
  assert.deepEqual(tensor.shape, [config.outputSize, config.inputSize]);
  assert.equal(config.inputSize, manifest.architecture.hiddenSize);
  manifest.artifactIdentity.manifestVariantId = config.manifestVariantId;
  const execution = manifest.inference.execution;
  const adapterModules = [];
  for (const declaration of config.kernels) {
    const bytes = await fs.readFile(path.join(config.kernelRoot, declaration.file));
    assert.equal(hash(bytes), declaration.sourceHash);
    assert(!execution.kernels[declaration.id], 'New mechanism ID must not replace an existing kernel');
    execution.kernels[declaration.id] = { kernel: declaration.file, entry: declaration.entry, digest: declaration.digest };
    execution.mechanismKernels.push(declaration.id);
    const artifactId = `adapter-kernel:${declaration.id}`;
    const artifact = { artifactId, role: 'wgsl-source', path: `adapter-kernels/${declaration.file}`,
      hash: declaration.sourceHash, sizeBytes: bytes.length };
    await fs.mkdir(path.dirname(path.join(output, artifact.path)), { recursive: true });
    await fs.writeFile(path.join(output, artifact.path), bytes, { flag: 'wx' });
    source.artifacts.push(artifact);
    source.wgslModules.push({ id: declaration.id, file: declaration.file, entry: declaration.entry,
      digest: declaration.digest, sourceHash: declaration.sourceHash, sourceArtifactId: artifactId });
    adapterModules.push(declaration.id);
  }
  assert(source.wgslModules.some(row => row.id === config.residualModule));
  // Use Forge's canonical entry-bound digest check before any physical load.
  await buildWgslClosure(execution, [], { repoRoot: config.kernelRoot, kernelSourceRoot: '.' });
  for (const artifact of parent.artifacts) {
    const destination = path.resolve(output, artifact.path);
    assert(destination.startsWith(output + path.sep), 'Artifact escapes candidate root');
    await fs.mkdir(path.dirname(destination), { recursive: true });
    if (artifact.artifactId === manifestArtifact.artifactId) continue;
    // Immutable sharing, not copying 8 GB. No code writes linked artifacts.
    await fs.link(path.join(root, artifact.path), destination);
  }
  await write(path.join(output, manifestArtifact.path), manifest);
  const newManifestBytes = await fs.readFile(path.join(output, manifestArtifact.path));
  manifestArtifact.hash = hash(newManifestBytes);
  manifestArtifact.sizeBytes = newManifestBytes.length;
  await write(path.join(output, 'source.json'), source);
  const weights = createZeroDeltaPeftFixture(config.inputSize, config.outputSize);
  const weightsPath = 'zero-delta.safetensors';
  await fs.writeFile(path.join(output, weightsPath), weights, { flag: 'wx' });
  const adapter = { id: config.adapterId, baseModel: parent.modelId, rank: 1, alpha: 2,
    targetModules: ['q_proj'], checksum: hash(weights), checksumAlgorithm: 'sha256',
    weightsFormat: 'safetensors', weightsPath, weightsSize: weights.length };
  await write(path.join(output, 'adapter-manifest.json'), adapter);
  const adapterExecution = { schema: 'doppler.capsule-adapter-execution/v1', maxAdapters: 1,
    combination: 'single', formats: ['peft_safetensors'], operations: ['generate'],
    kernelModules: [...adapterModules, config.residualModule] };
  await write(path.join(output, 'preparation.json'), { config, parentSha256: hash(parentBytes),
    manifestSha256: manifestArtifact.hash, adapterSha256: hash(weights), adapterExecution,
    qualification: false, signedCapsule: false, publication: false,
    claimBoundary: 'Controlled rank-one zero-delta PEFT fixture at layer 0 q_proj; not a trained adapter or nonzero adapter parity.' });
  assert.equal(hash(await fs.readFile(config.parentCapsule)), config.parentSha256);
  return { output, adapterExecution };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  console.log(JSON.stringify(await prepareAdapterEvaluationSource(await read(process.argv[2]))));
}
