import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import * as runtime from '../../src/partitions.js';
const hash='sha256:'+'a'.repeat(64), other='sha256:'+'b'.repeat(64);
assert.equal(typeof runtime.validateOrdinaryActivation,'function');
for (const [dtype,precision,components,stride] of [['f16','f16',1,2],['f32','f32',1,4],['f32','two-f32',2,8]]) {
  for (const sequence of [0,1]) {
    const rows=sequence===0?{start:0,end:3}:{start:3,end:4};
    const bytes=new Uint8Array((rows.end-rows.start)*4*stride);
    const frame={schema:'doppler.ordinary-activation/v1',attemptDigest:hash,sequence,baseCursorId:sequence===0?null:hash,
      proposalId:null,sourceStageId:hash,destinationStageId:other,rows,tokenContextDigest:hash,
      layout:{dtype,precision,components,elementStrideBytes:stride,hiddenSize:4,byteOrder:'little-endian'},
      payloadBytes:bytes.length,payloadDigest:'sha256:'+createHash('sha256').update(bytes).digest('hex')};
    const limits={maxRows:3,maxBytes:1024};
    const result=runtime.validateOrdinaryActivation(frame,bytes,limits);
    assert(Object.isFrozen(result.frame.layout));bytes[0]=7;assert.equal(result.bytes[0],0);bytes[0]=0;
    assert.throws(()=>runtime.validateBlockActivation(frame,bytes,limits),'ordinary frames never become block proposals');
    for (const mutate of [
      f=>{f.extra=true;},f=>{delete f.proposalId;},f=>{f.proposalId=hash;},f=>{f.schema='doppler.block-activation/v2';},
      f=>{f.baseCursorId=sequence===0?hash:null;},f=>{f.sequence=-1;},f=>{f.sequence=0.5;},
      f=>{f.layout.components=3;},f=>{f.layout.elementStrideBytes=stride*2;},f=>{f.layout.byteOrder='big-endian';},
      f=>{f.layout.hiddenSize=Number.MAX_SAFE_INTEGER;},f=>{f.payloadBytes++;},f=>{f.payloadDigest=hash;},
      f=>{f.sourceStageId=f.destinationStageId;},f=>{f.rows.start=-1;},f=>{f.rows.end=Number.MAX_SAFE_INTEGER;},
    ]) { const changed=structuredClone(frame);mutate(changed);assert.throws(()=>runtime.validateOrdinaryActivation(changed,bytes,limits)); }
    assert.throws(()=>runtime.validateOrdinaryActivation(frame,new Uint8Array(bytes.length+1),limits));
    assert.throws(()=>runtime.validateOrdinaryActivation(frame,bytes,{...limits,maxBytes:bytes.length-1}));
    if(sequence===0)assert.throws(()=>runtime.validateOrdinaryActivation({...frame,rows:{start:1,end:4}},bytes,limits));
    else assert.throws(()=>runtime.validateOrdinaryActivation({...frame,rows:{start:3,end:5}},new Uint8Array(8*stride),limits));
  }
}
console.log('ordinary activation: bootstrap/no proposal, exact scalar/paired bytes, immutable copy and strict malformed/overflow rejection pass');
