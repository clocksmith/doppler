// Artifact-fixture regression: production parser/compiler and the actual CLI.
// No model weights, device, source-string assertions or repository mutation.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';
import { parseManifest } from '../../src/formats/rdrr/index.js';
import { resolveRuntimeConfig } from '../../src/config/runtime.js';
import { applyExecutionV1RuntimeConfig, compileExecutionV1 } from '../../src/inference/pipelines/text/execution-v1.js';
import { getKernelPathAttentionVariant } from '../../src/config/kernel-path-loader.js';

const execute = promisify(execFile);
assert(process.argv[2], 'Pass the saved F32 input-config.json fixture');
const input = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
const manifest = parseManifest(await fs.readFile(path.join(input.rdrrDir, 'manifest.json'), 'utf8'));
const runtime = resolveRuntimeConfig(input.runtimeConfig);
const capabilities = { hasF16: true, hasSubgroups: true, maxWorkgroupStorageSize: 32768 };
const options = { manifestInference: manifest.inference, modelId: manifest.modelId,
  numLayers: manifest.architecture.numLayers, headDim: manifest.architecture.headDim,
  runtimeSession: runtime.inference.session, runtimeCompute: runtime.inference.compute,
  kernelPathPolicy: runtime.inference.kernelPathPolicy, capabilities, useGPU: true };
// This is the actual previously omitted argument, not an imitation of its code.
assert.throws(() => compileExecutionV1(options), /linear|projection/i);
const compiled = compileExecutionV1({ ...options, weightDtype: manifest.quantizationInfo?.weights ?? null });
const production = applyExecutionV1RuntimeConfig({ runtimeConfig: runtime, manifest, capabilities });
assert.deepEqual(compiled.runtimeInferencePatch, production.executionV1State.runtimeInferencePatch);

const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'attention-preparation-fixture-'));
try {
  const fixture = path.join(dir, 'model');
  await fs.mkdir(fixture);
  const fixtureInput = { ...input, rdrrDir: fixture };
  const inputPath = path.join(dir, 'input.json');
  await fs.writeFile(inputPath, JSON.stringify(fixtureInput));
  await fs.writeFile(path.join(fixture, 'manifest.json'), JSON.stringify(manifest));
  const cli = fileURLToPath(new URL('../../tools/experiments/prepare-f32-prefill-attention.js', import.meta.url));
  const output = path.join(dir, 'good');
  await execute(process.execPath, [cli, inputPath, output]);
  const receipt = JSON.parse(await fs.readFile(path.join(output, 'dry-run.json'), 'utf8'));
  assert(Object.values(receipt.gates).every(value => value === true));
  assert.equal(receipt.compileBlocker, null);
  assert.equal(receipt.routing.length, 10);
  const candidate = JSON.parse(await fs.readFile(path.join(output, 'runtime-candidate.json'), 'utf8'));
  const applied = applyExecutionV1RuntimeConfig({ runtimeConfig: resolveRuntimeConfig(candidate), manifest, capabilities });
  for (const { layer } of receipt.routing) {
    assert.equal(getKernelPathAttentionVariant('prefill', layer, applied.runtimeConfig.inference.kernelPath), 'prefill_query_tile_f32');
  }
  const missingWeight = structuredClone(manifest);
  delete missingWeight.quantizationInfo.weights;
  await fs.writeFile(path.join(fixture, 'manifest.json'), JSON.stringify(missingWeight));
  await assert.rejects(() => execute(process.execPath, [cli, inputPath, path.join(dir, 'bad')]));
  await assert.rejects(() => fs.stat(path.join(dir, 'bad', 'dry-run.json')), { code: 'ENOENT' });
  const badSemantic = structuredClone(manifest);
  badSemantic.inference.execution.prefill.push([
    'attention_prefill_query_tile_f32', Object.keys(badSemantic.inference.execution.kernels)[0],
  ]);
  assert.throws(() => parseManifest(JSON.stringify(badSemantic)), /unknown ops/i);
} finally {
  await fs.rm(dir, { recursive: true, force: true });
}
console.log('attention-f32-prefill-preparation: production compile, dtype omission and manifest schema fixtures passed');
