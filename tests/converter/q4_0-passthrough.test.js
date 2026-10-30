import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';
import { convertModel } from '../../src/converter/core.js';
import { createConverterConfig,DEFAULT_MANIFEST_INFERENCE } from '../../src/config/schema/index.js';
import { packedFixture } from '../helpers/q4_0-fixture.js';
installNodeFileFetchShim();
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const data=new Map(),tensors=[];
for(const [i,dtype] of ['Q4_0','Q4_1','Q8_0','Q6_K'].entries()) {
 const name=`model.layers.0.self_attn.${['q_proj','k_proj','v_proj','o_proj'][i]}.weight`;
 const bytes=packedFixture(['q4_0','q4_1','q8_0','q6k'][i],3,256);data.set(name,bytes);
 tensors.push({name,shape:[3,256],dtype,size:bytes.length,offset:0,layout:'row'});
}
const norm='model.layers.0.input_layernorm.weight',normBytes=new Uint8Array(Float32Array.from({length:256},(_,i)=>1+i*0.00001).buffer);
data.set(norm,normBytes);tensors.push({name:norm,shape:[256],dtype:'F32',size:normBytes.length,offset:0});
const shards=[];let manifest;
await convertModel({name:'q4-0-passthrough',quantization:'Q4_0',tensors,config:{model_type:'llama',eos_token_id:1}}, {
 async readTensorData(tensor){return data.get(tensor.name);},
 async writeShard(index,bytes){shards[index]=Uint8Array.from(bytes);return hash(bytes);},
 async writeManifest(value){manifest=value;}
}, {modelId:'q4-0-passthrough',modelType:'transformer',quantization:'Q4_0',
 quantizationInfo:{weights:'q4_0',embeddings:'f32',compute:'f32',preserveSourceTensors:true},
 architecture:{numLayers:1,hiddenSize:256,intermediateSize:1024,numAttentionHeads:4,numKeyValueHeads:4,headDim:64,vocabSize:256,maxSeqLen:128,ropeTheta:1000000},
 inference:{...DEFAULT_MANIFEST_INFERENCE},eosTokenId:1,converterConfig:createConverterConfig()});
for(const tensor of tensors) {
 const location=manifest.tensors[tensor.name], spans=location.spans??[{shardIndex:location.shard,offset:location.offset,size:location.size}];
 const bytes=Buffer.concat(spans.map(span=>Buffer.from(shards[span.shardIndex].subarray(span.offset,span.offset+span.size))));
 assert.equal(location.dtype,tensor.dtype);assert.equal(hash(bytes),hash(data.get(tensor.name)));assert.deepEqual(bytes,Buffer.from(data.get(tensor.name)));
}
console.log(JSON.stringify({status:'passed',tensors:tensors.length,scope:'Actual converter shards preserve original packed and floating source bytes'}));
