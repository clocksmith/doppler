import assert from 'node:assert/strict';
import { createLayerPartitionPlan } from '../../src/partitions.js';
import { executePartitionLayers } from '../../src/inference/pipelines/text/partition-execution.js';

const manifest = { modelId: 'recurrent', architecture: { numLayers: 4, hiddenSize: 8, vocabSize: 128 } };
const plan = createLayerPartitionPlan({ modelId: manifest.modelId, ...manifest.architecture, activationDtype: 'f32' });
const state = { manifest, useGPU: true, modelPartition: { plan, index: 1 }, currentSeqLen: 12,
  modelConfig: { useMoE: false, numKvSharedLayers: 0, hiddenSizePerLayerInput: null,
    decodeStrategy: 'incremental', causalAttention: true,
    layerTypes: ['linear_attention', 'full_attention', 'linear_attention', 'full_attention'] },
  runtimeConfig: { inference: { session: { usePostFfnNextInputRMSNormPairFusion: false } } },
  executionPlanState: { primaryPlan: { activationDtype: 'f32', finitenessGuardEnabled: false } },
  linearAttentionRuntime: { layers: new Map() } };
const signal = new AbortController().signal;
await assert.rejects(executePartitionLayers(state, { numTokens: 1 }, signal), /recurrent state missing.*layer 2/);
state.linearAttentionRuntime.layers.set(2, { seqLen: 11 });
await assert.rejects(executePartitionLayers(state, { numTokens: 1 }, signal), /recurrent state missing.*layer 2/);
assert.equal(state.linearAttentionRuntime.layers.get(2).seqLen, 11, 'Invalid continuation must never reset or fabricate state');
console.log('partition-recurrent-continuation: missing and mismatched state rejected before dispatch');
