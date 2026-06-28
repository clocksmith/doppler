import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { runTrainingOperatorCommand } from '../../src/experimental/training/operator-command.js';

const tmpRoot = await mkdtemp(join(tmpdir(), 'doppler-distill-sft-lora-'));
const datasetPath = join(tmpRoot, 'code-agent-sft.jsonl');
await writeFile(datasetPath, [
  JSON.stringify({
    rowId: 'js-review-1',
    prompt: 'system: Review JavaScript changes.\n\nuser: diff -- src/tooling/example.js',
    completion: 'assistant: finding high src/tooling/example.js:12 missing null handling; add a guard and a regression test.',
  }),
  JSON.stringify({
    rowId: 'wgsl-patch-1',
    prompt: 'system: Patch WGSL kernels.\n\nuser: storage buffer length may be zero.',
    completion: 'assistant: emit a bounds guard before reading the storage buffer and keep dispatch constants unchanged.',
  }),
].join('\n'), 'utf8');

function tensor(name, shape, seed) {
  const data = [];
  for (let index = 0; index < shape[0] * shape[1]; index += 1) {
    data.push((seed + index + 1) / 100);
  }
  return { name, shape, data };
}

const trainerPath = join(tmpRoot, 'trainer.js');
await writeFile(trainerPath, `
export function trainCausalLmLora(input) {
  return {
    checkpointId: 'checkpoint-sft',
    checkpointStep: input.training.steps,
    trainerId: 'distill-sft-test-provider',
    runnerId: 'distill-sft-lora',
    metrics: { trainLoss: 0.1875 },
    receipts: [{ backend: 'test-provider', datasetHash: input.dataset.datasetHash }],
    tensors: [
      ${JSON.stringify(tensor('layers.0.q_proj.lora_a', [3, 2], 0))},
      ${JSON.stringify(tensor('layers.0.q_proj.lora_b', [2, 4], 10))},
      ${JSON.stringify(tensor('layers.0.v_proj.lora_a', [3, 2], 20))},
      ${JSON.stringify(tensor('layers.0.v_proj.lora_b', [2, 4], 30))}
    ]
  };
}
`, 'utf8');

const workloadPath = join(tmpRoot, 'distill-sft.workload.json');
await writeFile(workloadPath, JSON.stringify({
  schemaVersion: 1,
  kind: 'distill',
  id: 'distill-sft-lora-bridge',
  description: 'SFT distillation bridge into causal-LM LoRA.',
  claimBoundary: 'Fixture proves distill SFT operator routing into LoRA adapter export.',
  seed: 1337,
  baseModelId: 'gemma4-e2b-it',
  studentModelId: 'gemma4-e2b-it',
  teacherModelId: 'zai-org/GLM-5.2',
  datasetId: 'code-agent-sft',
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
    steps: 1,
    precision: {
      activations: 'f16',
      gradients: 'f16',
      loraParams: 'f32',
    },
    gradientClipping: {
      maxNorm: 1,
    },
  },
  distill: {
    stagePlan: [{
      id: 'sft_lora',
      trainingStage: 'sft',
      objective: 'sft',
      steps: 1,
      checkpointEvery: 1,
      selectionMetric: 'loss',
      selectionGoal: 'min',
      evalSchedule: 'final',
    }],
    studentGraphMode: 'causal_lm_lora',
    temperature: 1,
    alphaKd: 0,
    alphaCe: 1,
    tripletMargin: 0.2,
    sourceLangs: [],
    targetLangs: [],
    pairAllowlist: [],
    strictPairContract: false,
    sftLora: {
      datasetFormat: 'text-pairs',
      taskType: 'text_generation',
      maxLength: 128,
      joinWith: '\n',
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
        id: 'distill-sft-lora-bridge',
        name: 'Distill SFT LoRA Bridge',
        format: 'manifest_json',
      },
      activation: {
        enabled: false,
        autoActivate: false,
        smokePrompt: null,
      },
      trainer: {
        modulePath: './trainer.js',
        exportName: 'trainCausalLmLora',
        runnerId: 'distill-sft-lora',
      },
    },
  },
}, null, 2), 'utf8');

const result = await runTrainingOperatorCommand({
  command: 'distill',
  action: 'run',
  workloadPath,
  runRoot: join(tmpRoot, 'run'),
  timestamp: '2026-06-27T00:00:00.000Z',
});

assert.equal(result.ok, true);
assert.equal(result.kind, 'distill');
assert.equal(result.stageResults.length, 1);
assert.equal(result.stageResults[0].trainingStage, 'sft');
assert.equal(result.stageResults[0].objective, 'sft');
assert.equal(result.stageResults[0].runnerKind, 'causal_lm_lora');
assert.equal(result.stageResults[0].exports.length, 1);
assert.equal(result.qualityGate.passed, true);

const stageManifest = JSON.parse(await readFile(result.stageResults[0].stageManifestPath, 'utf8'));
assert.equal(stageManifest.artifactType, 'distill_stage_manifest');
assert.equal(stageManifest.trainingStage, 'sft');
assert.equal(stageManifest.sftLoraRun.preflight.runnerKey, 'gemma4-e2b-it::text-pairs::text_generation');
assert.equal(stageManifest.exports.length, 1);

const adapterManifest = JSON.parse(await readFile(result.stageResults[0].exports[0].manifestPath, 'utf8'));
assert.equal(adapterManifest.baseModel, 'gemma4-e2b-it');
assert.equal(adapterManifest.metadata.runnerId, 'distill-sft-lora');
assert.equal(adapterManifest.metadata.trainerId, 'distill-sft-test-provider');

console.log('distill-sft-lora-bridge.test: ok');
