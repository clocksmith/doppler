import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { scopePipelineShaders } from '../../src/inference/pipelines/shader-scoped-pipeline.js';
import { createLayerPartitionPlan, hashLayerPartitionPlan } from '../../src/inference/pipelines/text/layer-partition-contract.js';

// Readiness ownership only. Enumeration and real pipeline creation are covered
// by resident-pipeline-preparation.test.js.
const target = new URL('../../src/inference/pipelines/text/resident-partition.js', import.meta.url).href;
const helper = './resident-pipeline-preparation.js';
const calls = [];
let resolvePreparation, rejectPreparation;
globalThis.residentPreparation = (...args) => { calls.push(args); return new Promise((resolve, reject) => { resolvePreparation = resolve; rejectPreparation = reject; }); };
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === target && specifier === helper) return { shortCircuit: true,
    url: 'data:text/javascript,export const prepareResidentPartitionPipelines=(...args)=>globalThis.residentPreparation(...args)' };
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
const receipt = { schema: 'doppler.resident-pipeline-preparation/v1', complete: true, count: 1, durationMs: 1,
  pipelines: [{ key: 'attention_paired.wgsl#main', operation: 'attention', variant: 'paired',
    shaderFile: 'attention_paired.wgsl', entryPoint: 'main', constants: null, roles: ['attention'] }], lazy: [] };
async function flush() { for (let i = 0; i < 12; i++) await Promise.resolve(); }
try {
  for (const disposition of ['success', 'failure', 'abort']) {
    const { pipeline, allocation } = fixture(); const controller = new AbortController();
    let settled = false;
    const opening = createResidentPartitionSession(pipeline, allocation, {}, async () => {}, controller.signal);
    opening.then(() => { settled = true; }, () => { settled = true; });
    await flush();
    assert.equal(settled, false, 'resident must not advertise ready while its pipelines are pending');
    assert.equal(calls.length, ['success', 'failure', 'abort'].indexOf(disposition) + 1);
    assert.equal(calls.at(-1)[0], pipeline, 'preparation runs on the loaded owner');
    assert.equal(calls.at(-1)[1], allocation, 'preparation enumerates the accepted allocation');
    assert.equal(calls.at(-1)[2], controller.signal, 'preparation receives the opening signal');
    if (disposition === 'abort') { controller.abort(Error('cancel preparation')); await flush(); assert.equal(settled, false, 'cancellation must await the already-started provider preparation'); }
    if (disposition === 'failure') { rejectPreparation(Error('compiler rejected')); await assert.rejects(opening, /compiler rejected/); }
    else { resolvePreparation(receipt); if (disposition === 'abort') await assert.rejects(opening, /cancel preparation/);
      else {
        const resident = await opening; const descriptor = resident.getDescriptor();
        assert.equal(descriptor.ready, true);
        assert.deepEqual(descriptor.pipelinePreparation, receipt, 'readiness reports the preparation receipt');
        descriptor.pipelinePreparation.pipelines.length = 0;
        assert.equal(resident.getDescriptor().pipelinePreparation.pipelines.length, 1, 'descriptor copies cannot alter the receipt');
        await resident.close();
      } }
  }
  // Every stage prepares its own program, including one without linear layers.
  const { pipeline, allocation } = fixture(undefined, 1);
  const opening = createResidentPartitionSession(pipeline, allocation, {}, async () => {});
  await flush(); assert.equal(calls.length, 4); assert.equal(calls.at(-1)[2], undefined);
  resolvePreparation(receipt);
  const resident = await opening;
  assert.deepEqual(resident.getDescriptor().pipelinePreparation, receipt);
  await resident.close();
} finally { hooks.deregister(); delete globalThis.residentPreparation; }
console.log('resident preparation: readiness, rejection, abort settlement and receipt reporting passed');
