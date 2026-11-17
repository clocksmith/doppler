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
import { applyExecutionV1RuntimeConfig, compileExecutionV1 } from "../../src/inference/pipelines/text/execution/execution-v1.js";
import { getKernelPathAttentionVariant } from '../../src/config/kernel-path-loader.js';

const execute = promisify(execFile);
const capabilities = { hasF16: true, hasSubgroups: true, maxWorkgroupStorageSize: 32768 };

async function createSyntheticF32Fixture() {
  const fixtureRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'attention-f32-synth-'));
  const rdrrDir = path.join(fixtureRoot, 'rdrr');
  await fs.mkdir(rdrrDir, { recursive: true });
  const base = JSON.parse(await fs.readFile('models/local/qwen-3-5-2b-q4k-ehaf16/manifest.json', 'utf8'));
  const sourceConfig = JSON.parse(await fs.readFile('src/config/conversion/qwen3/qwen-3-6-35b-a3b-q4_0-passthrough.json', 'utf8'));
  base.modelId = 'qwen-3-6-35b-a3b-text-source-q4_0-af32-fixture';
  base.architecture.numLayers = sourceConfig.inference.layerPattern.layerTypes.length;
  base.quantization = 'Q4_0';
  base.quantizationInfo = {
    weights: sourceConfig.quantization.weights,
    embeddings: sourceConfig.quantization.embeddings,
    lmHead: sourceConfig.quantization.lmHead,
    compute: sourceConfig.quantization.computePrecision,
    layout: sourceConfig.quantization.q4kLayout,
    variantTag: 'q4_0-af32',
  };
  base.inference = {
    ...structuredClone(base.inference),
    schema: 'doppler.execution/v1',
    attention: structuredClone(sourceConfig.inference.attention),
    normalization: structuredClone(sourceConfig.inference.normalization),
    ffn: structuredClone(sourceConfig.inference.ffn),
    rope: structuredClone(sourceConfig.inference.rope),
    output: structuredClone(sourceConfig.inference.output),
    layerPattern: structuredClone(sourceConfig.inference.layerPattern),
    chatTemplate: structuredClone(sourceConfig.inference.chatTemplate),
    session: structuredClone(sourceConfig.session),
    execution: structuredClone(sourceConfig.execution),
  };
  await fs.writeFile(path.join(rdrrDir, 'manifest.json'), `${JSON.stringify(base, null, 2)}\n`);
  const parsed = parseManifest(await fs.readFile(path.join(rdrrDir, 'manifest.json'), 'utf8'));
  const baseRuntime = resolveRuntimeConfig({
    inference: {
      session: {
        ...structuredClone(parsed.inference.session),
        kvcache: { ...structuredClone(parsed.inference.session.kvcache), kvDtype: 'f32' },
        compute: { defaults: { activationDtype: 'f32', mathDtype: 'f32', accumDtype: 'f32', outputDtype: 'f32' } },
      },
      kernelPathPolicy: { mode: 'capability-aware', sourceScope: ['model', 'manifest', 'config'], allowSources: ['model', 'manifest', 'config'], onIncompatible: 'remap' },
    },
  });
  const applied = applyExecutionV1RuntimeConfig({ runtimeConfig: baseRuntime, manifest: parsed, capabilities });
  const kernelPath = structuredClone(applied.runtimeConfig.inference.kernelPath);
  const coveredLayers = new Set((kernelPath.layerOverrides ?? []).flatMap((override) => override.layers));
  kernelPath.layerOverrides = [
    ...(kernelPath.layerOverrides ?? []),
    ...Array.from({ length: parsed.architecture.numLayers }, (_, layerIdx) => layerIdx)
      .filter((layerIdx) => !coveredLayers.has(layerIdx))
      .map((layerIdx) => ({
        layers: [layerIdx],
        decode: { steps: structuredClone(kernelPath.decode.steps) },
        prefill: { steps: structuredClone(kernelPath.prefill.steps) },
      })),
  ];
  for (const override of kernelPath.layerOverrides) {
    if (!override.prefill?.steps?.length) {
      override.prefill = { steps: structuredClone(kernelPath.prefill.steps) };
    }
    if (!override.decode?.steps?.length) {
      override.decode = { steps: structuredClone(kernelPath.decode.steps) };
    }
  }
  const inputConfigPath = path.join(fixtureRoot, 'input-config.json');
  await fs.writeFile(inputConfigPath, `${JSON.stringify({
    rdrrDir,
    runtimeConfig: {
      inference: {
        session: applied.runtimeConfig.inference.session,
        compute: applied.runtimeConfig.inference.compute,
        kernelPath,
      },
    },
  }, null, 2)}\n`);
  return { fixtureRoot, inputConfigPath };
}

const synthetic = process.argv[2] ? null : await createSyntheticF32Fixture();
const inputConfigArg = process.argv[2] ?? synthetic.inputConfigPath;
const input = JSON.parse(await fs.readFile(inputConfigArg, 'utf8'));
const manifest = parseManifest(await fs.readFile(path.join(input.rdrrDir, 'manifest.json'), 'utf8'));
const runtime = resolveRuntimeConfig(input.runtimeConfig);
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
  if (synthetic) await fs.rm(synthetic.fixtureRoot, { recursive: true, force: true });
}
console.log('attention-f32-prefill-preparation: production compile, dtype omission and manifest schema fixtures passed');
