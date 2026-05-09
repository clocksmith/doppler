import assert from 'node:assert/strict';
import { mkdtemp, readFile, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { runLoraPipeline } from '../../src/experimental/training/lora-pipeline.js';

const datasetText = [
  JSON.stringify({
    rowId: 'dream-row-1',
    source: 'system: Rewrite intent.\n\nuser: alpha',
    target: 'assistant: alpha rewrite',
  }),
  JSON.stringify({
    rowId: 'dream-row-2',
    source: 'system: Rewrite intent.\n\nuser: beta',
    target: 'assistant: beta rewrite',
  }),
].join('\n');

function tensor(name, shape, seed) {
  const data = new Float32Array(shape[0] * shape[1]);
  for (let index = 0; index < data.length; index += 1) {
    data[index] = (seed + index + 1) / 100;
  }
  return {
    name,
    shape,
    tensor: data,
  };
}

const tmpRoot = await mkdtemp(join(tmpdir(), 'doppler-causal-lm-runner-'));
const datasetPath = join(tmpRoot, 'rows.jsonl');
await writeFile(datasetPath, datasetText, 'utf8');

const workload = {
  schemaVersion: 1,
  kind: 'lora',
  id: 'dream-causal-lm-runner',
  description: 'Dream causal-LM LoRA runner contract fixture',
  claimBoundary: 'Fixture proves provider-backed runner/export wiring only.',
  seed: 7,
  baseModelId: 'gemma4-e2b-it',
  datasetId: 'dream-text-pairs',
  datasetPath,
  evalDatasets: [],
  trainingSchemaVersion: 1,
  checkpointEvery: 1,
  selectionMetric: 'loss',
  selectionGoal: 'min',
  surfaceSupport: 'node',
  training: {
    optimizer: {
      type: 'adamw',
      lr: 0.0001,
      beta1: 0.9,
      beta2: 0.999,
      eps: 1e-8,
      weightDecay: 0,
      scheduler: {
        enabled: false,
        type: 'constant',
        warmupSteps: 0,
        stepSize: 1,
        gamma: 1,
        totalSteps: 1,
        minLr: 0,
      },
    },
    batchSize: 1,
    accumSteps: 1,
    steps: 3,
    precision: {
      activations: 'f16',
      gradients: 'f16',
      loraParams: 'f32',
    },
    gradientClipping: {
      maxNorm: 1,
    },
  },
  pipeline: {
    datasetFormat: 'text-pairs',
    taskType: 'text_generation',
    adapter: {
      rank: 2,
      alpha: 4,
      dropout: 0,
      targetModules: ['q_proj', 'v_proj'],
    },
    freeze: {
      encoder: false,
      prior: false,
      decoder: false,
      base: true,
      lora: false,
    },
    export: {
      enabled: true,
      atCheckpoints: false,
      select: 'final',
      id: 'dream-causal-lm-fixture',
      name: 'Dream causal-LM fixture',
      format: 'manifest_json',
    },
    activation: null,
  },
  configHash: 'sha256:test',
};

const result = await runLoraPipeline({
  loadedWorkload: {
    absolutePath: join(tmpRoot, 'dream-causal-lm-runner.workload.json'),
    path: join(tmpRoot, 'dream-causal-lm-runner.workload.json'),
    raw: JSON.stringify(workload),
    workloadSha256: 'sha256:test',
    workload,
  },
  runRoot: join(tmpRoot, 'run'),
  causalLmTrainer: async (input) => {
    assert.equal(input.runnerKind, 'causal_lm_lora');
    assert.equal(input.preflight.supported, true);
    assert.equal(input.dataset.rowCount, 2);
    assert.equal(input.adapter.rank, 2);
    return {
      checkpointId: 'checkpoint-final',
      checkpointStep: input.training.steps,
      trainerId: 'test-provider',
      runnerId: 'test-causal-lm-lora',
      metrics: {
        trainLoss: 0.125,
      },
      receipts: [{
        backend: 'test-provider',
        datasetHash: input.dataset.datasetHash,
      }],
      tensors: [
        tensor('layers.0.q_proj.lora_a', [3, 2], 0),
        tensor('layers.0.q_proj.lora_b', [2, 4], 10),
        tensor('layers.0.v_proj.lora_a', [3, 2], 20),
        tensor('layers.0.v_proj.lora_b', [2, 4], 30),
      ],
    };
  },
});

assert.equal(result.ok, true);
assert.equal(result.runnerKind, 'causal_lm_lora');
assert.equal(result.preflight.runnerKey, 'gemma4-e2b-it::text-pairs::text_generation');
assert.equal(result.exports.length, 1);
assert.match(result.exports[0].weightsSha256, /^[a-f0-9]{64}$/);

const exportManifest = JSON.parse(await readFile(result.exports[0].manifestPath, 'utf8'));
const runtimeManifest = JSON.parse(await readFile(result.exports[0].runtimeManifestPath, 'utf8'));
assert.equal(exportManifest.baseModel, 'gemma4-e2b-it');
assert.equal(runtimeManifest.id, exportManifest.id);
assert.equal(exportManifest.weightsFormat, 'safetensors');
assert.equal(exportManifest.metadata.runnerKind, 'causal_lm_text_generation');
assert.equal(exportManifest.metadata.runnerId, 'test-causal-lm-lora');
assert.equal(exportManifest.metadata.trainerId, 'test-provider');

const weightsStats = await stat(result.exports[0].weightsPath);
assert.equal(weightsStats.size, exportManifest.weightsSize);

const trainerModulePath = join(tmpRoot, 'fixture-trainer.js');
await writeFile(trainerModulePath, `
export function trainCausalLmLora(input) {
  return {
    checkpointId: 'checkpoint-module',
    checkpointStep: input.training.steps,
    trainerId: 'module-provider',
    runnerId: 'module-causal-lm-lora',
    metrics: { trainLoss: 0.25 },
    tensors: [
      { name: 'layers.0.q_proj.lora_a', shape: [3, 2], data: [0.01, 0.02, 0.03, 0.04, 0.05, 0.06] },
      { name: 'layers.0.q_proj.lora_b', shape: [2, 4], data: [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18] },
      { name: 'layers.0.v_proj.lora_a', shape: [3, 2], data: [0.21, 0.22, 0.23, 0.24, 0.25, 0.26] },
      { name: 'layers.0.v_proj.lora_b', shape: [2, 4], data: [0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38] }
    ]
  };
}
`, 'utf8');

const moduleWorkload = {
  ...workload,
  id: 'dream-causal-lm-module-runner',
  pipeline: {
    ...workload.pipeline,
    export: {
      ...workload.pipeline.export,
      id: 'dream-causal-lm-module-fixture',
    },
    trainer: {
      modulePath: './fixture-trainer.js',
      exportName: 'trainCausalLmLora',
      runnerId: 'module-causal-lm-lora',
    },
  },
};

const moduleResult = await runLoraPipeline({
  loadedWorkload: {
    absolutePath: join(tmpRoot, 'module.workload.json'),
    path: join(tmpRoot, 'module.workload.json'),
    raw: JSON.stringify(moduleWorkload),
    workloadSha256: 'sha256:module-test',
    workload: moduleWorkload,
  },
  runRoot: join(tmpRoot, 'module-run'),
});

assert.equal(moduleResult.ok, true);
assert.equal(moduleResult.exports.length, 1);
const moduleManifest = JSON.parse(await readFile(moduleResult.exports[0].manifestPath, 'utf8'));
assert.equal(moduleManifest.metadata.runnerId, 'module-causal-lm-lora');
assert.equal(moduleManifest.metadata.trainerId, 'module-provider');

console.log('lora-causal-lm-provider-runner.test: ok');
