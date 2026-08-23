import assert from 'node:assert/strict';
import test from 'node:test';

import {
  bindAttentionPlan,
  createAttentionExecutor,
  executeBoundAttentionPlan,
  resolveAttentionPlan,
} from '../../src/inference/pipelines/text/attention/plan.js';
import {
  createRefactorReceipt,
  verifyRefactorReceipt,
} from '../../src/inference/refactor-receipt.js';

function createPlanInput() {
  return {
    phase: 'decode',
    geometry: {
      layerIdx: 3,
      numTokens: 1,
      numHeads: 8,
      numKVHeads: 2,
      headDim: 128,
      hiddenSize: 1024,
      currentSeqLen: 12,
    },
    dtypes: {
      input: 'f32',
      activation: 'f32',
      projection: 'f32',
      kv: 'f16',
      cachedK: 'f16',
      cachedV: 'f16',
      outputProjectionInput: 'f32',
      output: 'f32',
    },
    kv: {
      layout: 'contiguous',
      coldQuantMode: 'none',
      length: 13,
      pageSize: 0,
    },
    session: {
      useWideTileResidualFusion: false,
      retainQ4KMaterialization: false,
    },
    capabilities: {
      hasF16: true,
      hasSubgroups: true,
    },
    fusion: {
      inputNormProjection: true,
      qkvProjection: true,
      qkNorm: false,
      qkNormRoPE: false,
    },
    queryTransform: {
      scale: 1,
      rope: 'enabled',
    },
    outputGate: {
      enabled: false,
      tensorDtype: null,
      tensorElementCount: 0,
      kernelPathSupportsFusion: false,
    },
    outputProjection: {
      weightDtype: 'f16',
      hasResidual: true,
      residualDtype: 'f32',
      hasLoRA: false,
    },
    diagnosticsEligible: false,
    sharedKV: {
      reuses: false,
      stores: false,
    },
    diffusionGemmaDecoder: false,
  };
}

test('immediate and recorded modes consume the identical semantic plan', async () => {
  const immediatePlan = resolveAttentionPlan(createPlanInput());
  const recordedPlan = resolveAttentionPlan(createPlanInput());

  assert.equal(immediatePlan.id, recordedPlan.id);
  assert.deepEqual(immediatePlan, recordedPlan);
  assert.equal(JSON.stringify(immediatePlan).includes('GPU'), false);

  const immediateStages = [];
  const recordedStages = [];
  const immediate = bindAttentionPlan(immediatePlan, {});
  const recorded = bindAttentionPlan(recordedPlan, {});
  await executeBoundAttentionPlan(immediate, createAttentionExecutor('immediate', {
    attention: () => immediateStages.push('attention'),
  }));
  await executeBoundAttentionPlan(recorded, createAttentionExecutor('recorded', {
    attention: () => recordedStages.push('attention'),
  }));
  assert.deepEqual(immediateStages, recordedStages);
});

test('changing one semantic decision changes the plan identity', () => {
  const baseline = resolveAttentionPlan(createPlanInput());
  const changedInput = createPlanInput();
  changedInput.kv.layout = 'tiered';
  const changed = resolveAttentionPlan(changedInput);

  assert.notEqual(baseline.id, changed.id);
  assert.notEqual(baseline.attention.implementation, changed.attention.implementation);
});

test('query scaling and NoPE are immutable plan identity', () => {
  const baseline = resolveAttentionPlan(createPlanInput());
  const changedInput = createPlanInput();
  changedInput.queryTransform.scale = 3.87;
  changedInput.queryTransform.rope = 'disabled';
  const changed = resolveAttentionPlan(changedInput);

  assert.notEqual(baseline.id, changed.id);
  assert.deepEqual(changed.queryTransform, { scale: 3.87, rope: 'disabled' });
  assert.throws(() => resolveAttentionPlan({
    ...createPlanInput(),
    queryTransform: { scale: 0, rope: 'enabled' },
  }), /positive finite number/);
});

test('semantic plans are deeply immutable and JSON-safe', () => {
  const plan = resolveAttentionPlan(createPlanInput());
  assert.doesNotThrow(() => JSON.stringify(plan));
  assert.equal(Object.isFrozen(plan), true);
  assert.equal(Object.isFrozen(plan.outputProjection), true);
  assert.throws(() => {
    plan.outputProjection.residualMode = 'separate';
  }, TypeError);
});

test('characterization matrix covers phase, dtype, capability, KV, fusion, LoRA, gate, and failure contracts', () => {
  const cases = [
    {
      id: 'gemma-decode-f32-contiguous',
      mutate() {},
    },
    {
      id: 'gemma-prefill-f16-paged',
      mutate(input) {
        input.phase = 'prefill';
        input.geometry.numTokens = 16;
        input.geometry.currentSeqLen = 0;
        for (const key of Object.keys(input.dtypes)) {
          input.dtypes[key] = 'f16';
        }
        input.kv.layout = 'paged';
        input.kv.length = 16;
        input.kv.pageSize = 16;
        input.session.useFlashPrefillAttention = true;
        input.fusion.qkNorm = true;
        input.fusion.qkNormRoPE = true;
        input.outputProjection.hasResidual = false;
        input.outputProjection.residualDtype = null;
        input.diagnosticsEligible = true;
      },
    },
    {
      id: 'qwen-decode-tiered-int8-lora',
      mutate(input) {
        input.kv.layout = 'tiered';
        input.kv.coldQuantMode = 'int8';
        input.kv.length = 64;
        input.fusion.qkvProjection = false;
        input.outputProjection.hasLoRA = true;
        input.sharedKV.reuses = true;
      },
    },
    {
      id: 'qwen-decode-output-gate-fused',
      mutate(input) {
        input.geometry.headDim = 256;
        input.geometry.hiddenSize = 2048;
        input.outputGate.enabled = true;
        input.outputGate.tensorDtype = 'f32';
        input.outputGate.tensorElementCount = 2048;
        input.outputGate.kernelPathSupportsFusion = true;
        input.session.attentionDecodeOnline = {
          useOutputGateFusion: true,
        };
      },
    },
    {
      id: 'llama-decode-no-f16-failure',
      failure: {
        boundary: 'output-projection',
        name: 'Error',
        message: 'fixture failure',
        cleanup: 'completed',
      },
      mutate(input) {
        input.capabilities.hasF16 = false;
        input.capabilities.hasSubgroups = false;
        input.dtypes.kv = 'f32';
        input.dtypes.cachedK = 'f32';
        input.dtypes.cachedV = 'f32';
        input.outputProjection.weightDtype = 'q4k';
        input.session.useWideTileResidualFusion = true;
        input.session.retainQ4KMaterialization = true;
        input.session.useWideTileQ4KDecode = true;
      },
    },
  ];

  const planIds = new Set();
  for (const characterization of cases) {
    const input = createPlanInput();
    characterization.mutate(input);
    const plan = resolveAttentionPlan(input);
    const repeated = resolveAttentionPlan(input);
    assert.deepEqual(plan, repeated, `${characterization.id} must be deterministic`);
    planIds.add(plan.id);

    const receipt = createRefactorReceipt({
      commandContext: {
        command: 'verify',
        workload: 'inference',
        intent: 'verify',
      },
      resolvedSession: {
        id: characterization.id,
        capabilities: input.capabilities,
        dtypes: input.dtypes,
      },
      observationContext: {
        diagnostics: input.diagnosticsEligible ? 'always' : 'on_failure',
        receiptPolicy: 'always',
      },
      operationPlan: plan,
      operations: plan.stages.map((stage, sequence) => ({ sequence, stage })),
      dtypeTransitions: Object.entries(plan.transitions)
        .filter(([, transition]) => transition !== null)
        .map(([boundary, transition]) => ({ boundary, transition })),
      resourceEvents: [
        { sequence: 0, action: 'acquire', label: 'query', ownership: 'scopeOwned' },
        { sequence: 1, action: 'release', label: 'query', ownership: 'scopeOwned' },
      ],
      failure: characterization.failure ?? null,
    });
    assert.equal(verifyRefactorReceipt(receipt), true);
  }

  assert.equal(planIds.size, cases.length);
});
