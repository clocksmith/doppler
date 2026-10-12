import assert from 'node:assert/strict';
import {createTensor,tensorBytes,dtypeBytes} from '../../src/gpu/tensor.js';
const representation={precision:'two-f32',components:2,elementStrideBytes:8};
const buffer={size:32};
const tensor=createTensor(buffer,'f32',[1,4],'paired',representation);
assert.equal(tensor.precision,'two-f32');assert.equal(tensor.components,2);assert.equal(tensor.elementStrideBytes,8);
assert.equal(tensorBytes(tensor.shape,tensor.dtype,tensor),32);
assert.equal(tensorBytes([1,4],'f32'),16,'weights and scalar tensors never widen from another tensor or session');
assert.equal(dtypeBytes('f32'),4);assert.equal(dtypeBytes('f16'),2);
const legacy=createTensor(buffer,'f32',[1,4],'legacy');
assert.deepEqual(legacy,{buffer,dtype:'f32',shape:[1,4],label:'legacy'});
for(const change of [{precision:'f32'},{components:1},{elementStrideBytes:4},{precision:undefined}]){
 assert.throws(()=>createTensor(buffer,'f32',[1,4],'bad',{...representation,...change}),/precision|component|stride|representation/);
}
for(const invalid of [null,'two-f32',2,[]]) assert.throws(()=>createTensor(buffer,'f32',[1,4],'bad',invalid),/representation/);
assert.throws(()=>createTensor(buffer,'f16',[1,4],'bad',representation),/f32|dtype/);
assert.throws(()=>tensorBytes([Number.MAX_SAFE_INTEGER,2],'f32',representation),/size|safe/);
assert.throws(()=>tensorBytes([1,-1],'f32',representation),/shape/);
assert.throws(()=>createTensor({size:16},'f32',[1,4],'short',representation),/buffer|bytes/);
assert.throws(()=>{tensor.components=1;},TypeError);
tensor.outputGateFused=true;assert.equal(tensor.outputGateFused,true,'unrelated operation annotation remains supported');
console.log('paired tensor metadata and explicit byte geometry passed');
