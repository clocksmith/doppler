import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { scopePipelineShaders } from '../../src/inference/pipelines/shader-scoped-pipeline.js';
import { createLayerPartitionPlan, hashLayerPartitionPlan } from '../../src/inference/pipelines/text/layer-partition-contract.js';

const target = new URL('../../src/inference/pipelines/text/resident-partition.js', import.meta.url).href;
const helper = '../../../gpu/kernels/paired-linear-attention.js';
let resolvePreparation, rejectPreparation, called = 0;
globalThis.residentPreparation = () => { called++; return new Promise((resolve, reject) => { resolvePreparation = resolve; rejectPreparation = reject; }); };
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === target && specifier === helper) return { shortCircuit: true,
    url: 'data:text/javascript,export const preparePairedLinearAttentionPipelines=(...args)=>globalThis.residentPreparation(...args)' };
  return next(specifier, context);
} });
const { createResidentPartitionSession } = await import(target);
function fixture(layerTypes = ['linear_attention', 'full_attention'], index = 0) {
  const plan = createLayerPartitionPlan({ modelId: 'synthetic', numLayers: 2, hiddenSize: 2, vocabSize: 8,
    activationPrecision: 'two-f32' });
  const pipeline = { useGPU: true, modelPartition: { plan, index }, kvCache: { destroy() {} },
    modelConfig: { layerTypes, useMoE: false, numKvSharedLayers: 0, hiddenSizePerLayerInput: null,
      decodeStrategy: 'incremental', causalAttention: true },
    executionPlanState: { primaryPlan: { activationDtype: 'f32', activationPrecision: 'two-f32',
      activationComponents: 2, activationElementStrideBytes: 8, finitenessGuardEnabled: false } },
    resolvedRuntimeSession: { runtime: { session: { intermediatePrecision: 'two-f32' } } },
    runtimeConfig: { inference: { session: { usePostFfnNextInputRMSNormPairFusion: false } } },
    gpuContext: null, dopplerLoader: { gpuBuffers: [] } };
  scopePipelineShaders(pipeline);
  const allocation = { model: { id: 'synthetic', identity: 'sha256:synthetic' }, plan,
    planId: hashLayerPartitionPlan(plan), index, generation: {} };
  return { pipeline, allocation };
}
async function flush() { for (let i = 0; i < 12; i++) await Promise.resolve(); }
try {
  for (const disposition of ['success', 'failure', 'abort']) {
    const { pipeline, allocation } = fixture(); const controller = new AbortController();
    let settled = false;
    const opening = createResidentPartitionSession(pipeline, allocation, {}, async () => {}, controller.signal);
    opening.then(() => { settled = true; }, () => { settled = true; });
    await flush();
    assert.equal(settled, false, 'resident must not advertise ready while its recurrent pipeline is pending');
    assert.equal(called, ['success', 'failure', 'abort'].indexOf(disposition) + 1);
    if (disposition === 'abort') { controller.abort(Error('cancel preparation')); await flush(); assert.equal(settled, false, 'cancellation must await the already-started provider preparation'); }
    if (disposition === 'failure') { rejectPreparation(Error('compiler rejected')); await assert.rejects(opening, /compiler rejected/); }
    else { resolvePreparation(); if (disposition === 'abort') await assert.rejects(opening, /cancel preparation/);
      else { const resident = await opening; assert.equal(resident.getDescriptor().ready, true); await resident.close(); } }
  }
  const { pipeline, allocation } = fixture(undefined, 1);
  const resident = await createResidentPartitionSession(pipeline, allocation, {}, async () => {});
  assert.equal(called, 3, 'a partition without assigned linear layers must not prepare unrelated recurrent pipelines');
  await resident.close();
} finally { hooks.deregister(); delete globalThis.residentPreparation; }
console.log('resident preparation: readiness, rejection, abort settlement and assigned-layer scope passed');
