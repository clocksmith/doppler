import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice,destroyDevice } from '../../src/gpu/device.js';
import { acquireBuffer,releaseBuffer,uploadData,readBuffer } from '../../src/memory/buffer-pool.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { runLinearAttentionCoreGPU } from '../../src/gpu/kernels/linear-attention-core.js';
const probe=await probeNodeGPU({installFileFetchShim:true});assert.ok(probe.ready,probe.reason);
const owned=[],D=4,NK=2,NV=4,H=32,headMap=[0,2,1,3];
const upload=values=>{const b=acquireBuffer(values.byteLength);uploadData(b,values);owned.push(b);return b;};
const values=(n,f)=>Float32Array.from({length:n},(_,i)=>Math.sin(i*f)*0.1);
const reorder=(array,width,rows,offset=0)=>{const out=Float32Array.from(array);for(let row=0;row<rows;row++)for(let h=0;h<NV;h++)for(let d=0;d<width;d++)out[row*(offset+NV*width)+offset+h*width+d]=array[row*(offset+NV*width)+offset+headMap[h]*width+d];return out;};
const conv=values(H*2,0.13),bias=values(NV,0.31);
const makeState=native=>({sourceLayout:native?'gguf-v1':'hf',convDim:H,convKernelSize:2,valueDim:NV*D,numVHeads:NV,numKHeads:NK,headKDim:D,headVDim:D,qSize:NK*D,kSize:NK*D,qRep:2,normMode:'shared',rmsNormEps:1e-6,
 convWeightGPU:upload(native?reorder(conv,D*2,1,2*NK*D*2):conv),dtBiasGPU:upload(native?reorder(bias,1,1):bias),aLogGPU:upload(new Float32Array(NV).fill(native?-1:0)),normWeightGPU:upload(new Float32Array(D).fill(1)),
 convStateGPU:upload(new Float32Array(H)),recurrentStateGPU:upload(new Float32Array(NV*D*D)),recurrentStatePrecision:'f32',recurrentStateComponents:1});
try {
 const standard=makeState(false),native=makeState(true),cases=[];
 for(const T of [3,1]) {
  const qkv=values(T*H,0.23),z=values(T*NV*D,0.17),a=values(T*NV,0.11),b=values(T*NV,0.29);
  const outputs=[];
  for(const [state,tiled] of [[standard,false],[native,true]]) {
   const tensor=(data,shape)=>createTensor(upload(data),'f32',shape);
   const output=await runLinearAttentionCoreGPU(tensor(tiled?reorder(qkv,D,T,2*NK*D):qkv,[T,H]),tensor(tiled?reorder(z,D,T):z,[T,NV*D]),tensor(tiled?reorder(a,1,T):a,[T,NV]),tensor(tiled?reorder(b,1,T):b,[T,NV]),state,
    {numTokens:T,normGateActivation:'silu',outputDtype:'f32',qkL2NormEps:1e-6,runtimeSession:{useLinearAttentionRegisterPrefill:false,useLinearAttentionRegisterDecode:false,useLinearAttentionFusedDecodeCore:false}});
   owned.push(output.buffer);outputs.push(new Float32Array(await readBuffer(output.buffer,T*NV*D*4)));
  }
  const expected=reorder(outputs[0],D,T);let maxAbs=0;for(let i=0;i<expected.length;i++){maxAbs=Math.max(maxAbs,Math.abs(expected[i]-outputs[1][i]));assert.ok(Number.isFinite(outputs[1][i])&&Math.abs(expected[i]-outputs[1][i])<0.00001);}
  cases.push({T,maxAbs});
 }
 console.log(JSON.stringify({status:'passed',sourceSemantics:'GGUF tiled V heads and negative-exponential A',cases}));
} finally {for(const b of owned)releaseBuffer(b);destroyDevice();}
