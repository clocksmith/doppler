import assert from 'node:assert/strict';
import { createLayerPartitionPlan, hashLayerPartitionPlan, resolveLayerPartition, serializeActivationFrame,
 deserializeActivationFrame, validateActivationTensorShape, getPartitionActivationContract, validateActivationFrameForPlan } from '../../src/inference/pipelines/text/layer-partition-contract.js';
const dimensions={modelId:'paired-fixture',numLayers:24,hiddenSize:1024,vocabSize:10};
const scalar=createLayerPartitionPlan(dimensions);
assert.equal(hashLayerPartitionPlan(scalar),'sha256:b6985918551275cd4f11f9cf7134cc4c5fbef5a10c2224e64bda607e782ebb4c');
const paired=createLayerPartitionPlan({...dimensions,activationPrecision:'two-f32'});
assert.equal(paired.schema,'doppler.layer-partition-contract/v2');
assert.equal(paired.activationPrecision,'two-f32');assert.equal(paired.activationComponents,2);assert.equal(paired.activationElementStrideBytes,8);
assert.notEqual(hashLayerPartitionPlan(paired),hashLayerPartitionPlan(scalar));
const manifest={modelId:dimensions.modelId,architecture:dimensions};
assert.deepEqual(resolveLayerPartition(manifest,{plan:paired,index:1}).layerRange,[12,23]);
assert.throws(()=>resolveLayerPartition(manifest,{plan:{...paired,activationComponents:1},index:1}),/match/);
assert.throws(()=>createLayerPartitionPlan({...dimensions,activationPrecision:'two-f32',activationDtype:'f16'}),/precision|f32/);
const layout={dtype:'f32',precision:'two-f32',components:2,elementStrideBytes:8};
// Exact interleaved bytes, including a subnormal low word and signed zero, are transport data, not JS tensor math.
const bits=new Uint32Array([0x3f800000,1,0x80000000,0]);const shape=[1,1,2];
const frame=serializeActivationFrame({...layout,shape,data:bits,step:3,seqOffset:9});
assert.equal(frame.schema,'doppler.activation-tensor/v2');assert.equal(frame.byteLength,16);
assert.deepEqual(new Uint32Array(frame.buffer),bits);
const decoded=deserializeActivationFrame(frame);assert.equal(decoded.precision,'two-f32');assert.equal(decoded.components,2);
assert.deepEqual(new Uint32Array(decoded.tensorData.buffer),bits);
assert.equal(validateActivationTensorShape({...layout,shape,byteLength:16}).expectedBytes,16);
for(const change of [{components:1},{elementStrideBytes:4},{dtype:'f16'},{precision:'f32'},{precision:undefined}]){
 assert.throws(()=>deserializeActivationFrame({...frame,...change}),/precision|component|stride|paired|f32/i);
}
assert.throws(()=>deserializeActivationFrame({...frame,schema:'doppler.activation-tensor/v1'}),/schema|precision|paired/i);
assert.throws(()=>deserializeActivationFrame({...frame,buffer:new ArrayBuffer(8)}),/byteLength/);
assert.throws(()=>deserializeActivationFrame({...frame,byteLength:8}),/byteLength/);
console.log('paired partition contract tests passed');

const smallPlan=createLayerPartitionPlan({...dimensions,hiddenSize:2,activationPrecision:'two-f32'});
assert.equal(getPartitionActivationContract(smallPlan).elementStrideBytes,8);
assert.deepEqual(new Uint32Array(validateActivationFrameForPlan(frame,smallPlan).tensorData.buffer),bits);
assert.throws(()=>validateActivationFrameForPlan(frame,scalar),/representation/);
assert.throws(()=>validateActivationFrameForPlan(frame,paired),/hiddenSize/);
assert.throws(()=>getPartitionActivationContract({...smallPlan,activationComponents:1}),/match/);
