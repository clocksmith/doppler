import assert from 'node:assert/strict';
import {createLayerPartitionPlan,hashLayerPartitionPlan} from '../../src/partitions.js';
import {createResidentPartitionSession} from '../../src/inference/pipelines/text/resident-partition.js';
import {scopePipelineShaders} from '../../src/inference/pipelines/shader-scoped-pipeline.js';
import {resolveGenerationOptions} from '../../src/config/generation-contract.js';
const plan=createLayerPartitionPlan({modelId:'paired',numLayers:2,hiddenSize:2,vocabSize:8,activationPrecision:'two-f32'});
const generation=resolveGenerationOptions({maxTokens:1,maxSeqLen:8,temperature:0,topK:1,topP:1,repetitionPenalty:1,repetitionPenaltyWindow:0,useChatTemplate:false});
let cacheClosures=0;
const pipeline={useGPU:true,modelPartition:{plan,index:1},gpuContext:null,dopplerLoader:{gpuBuffers:[]},
 kvCache:{destroy(){cacheClosures++;}},
 modelConfig:{useMoE:false,numKvSharedLayers:0,hiddenSizePerLayerInput:null,decodeStrategy:'incremental',causalAttention:true},
 executionPlanState:{primaryPlan:{activationDtype:'f32',activationPrecision:'two-f32',activationComponents:2,activationElementStrideBytes:8,finitenessGuardEnabled:false}},
 runtimeConfig:{inference:{session:{usePostFfnNextInputRMSNormPairFusion:false}}},
 resolvedRuntimeSession:{runtime:{session:{intermediatePrecision:'two-f32'}}}};
scopePipelineShaders(pipeline);
const allocation={model:{id:'paired',identity:'sha256:model'},plan,planId:hashLayerPartitionPlan(plan),index:1,participantId:'B',generation,
 limits:{maxTokens:1,maxPromptTokens:4,maxActivationBytes:32,maxOutputCharacters:100,maxAttempts:16,maxConcurrentAttempts:1}};
const session=await createResidentPartitionSession(pipeline,allocation,{},async()=>{});
assert.equal(cacheClosures,1);
let id=0;
const request=activation=>({identity:{modelId:'paired',modelIdentity:'sha256:model',planId:allocation.planId,threadId:'thread',attemptId:String(++id),participantA:'A',participantB:'B'},
 inputTokenIds:[1],inputTokenCount:1,step:0,tokenPosition:0,maxTokens:1,generation,continuation:null,signal:new AbortController().signal,activation});
const activation={shape:[1,1,2],dtype:'f32',precision:'two-f32',components:2,elementStrideBytes:8,step:0,seqOffset:0,tensorData:new Float32Array(4)};
for(const change of [{precision:undefined},{components:1},{elementStrideBytes:4},{dtype:'f16'},{tensorData:new Float32Array(2)}]){
 await assert.rejects(session.executeGroup1(request({...activation,...change})),/precision|layout|identity|byteLength/);
}
const over=request(activation);over.inputTokenCount=3;over.inputTokenIds=[1,2,3];
await assert.rejects(session.executeGroup1(over),/byte allocation/);
await session.close();
console.log('paired resident malformed input and full paired byte allowance rejected before attempt resource creation');
