import assert from 'node:assert/strict';
import { packedFixture, decodePacked, projectReference } from './q4_0-fixture.js';
import { getDevice } from '../../src/gpu/device.js';
import { acquireBuffer, uploadData, readBuffer, releaseBuffer } from '../../src/memory/buffer-pool.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { createCommandRecorder } from '../../src/gpu/command-recorder.js';
import { runQ40Matmul, recordQ40Matmul } from '../../src/gpu/kernels/q4_0.js';
import { executeSharedSlotGateUp, executeSharedSlotDown } from '../../src/gpu/kernels/moe-shared-slot.js';
import { runResidualAdd, recordResidualAdd } from '../../src/gpu/kernels/residual.js';


export async function runNativePortFixtures() {
const device = getDevice();
const allocated = new Set(), recorders = [];
const upload = data => {
  const padded = new Uint8Array(Math.ceil(data.byteLength / 4) * 4);
  padded.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  const buffer = acquireBuffer(padded.byteLength); allocated.add(buffer); uploadData(buffer, padded); return buffer;
};
const tensor = (data, shape) => createTensor(upload(data), 'f32', shape);
const packed = (dtype, shape) => {
  const K = shape.at(-1), rows = shape.slice(0, -1).reduce((a, b) => a*b, 1);
  const bytes = packedFixture(dtype, rows, K);
  return { weight: createWeightBuffer(upload(bytes), dtype, 'row', shape), decoded: decodePacked(dtype, bytes, rows, K) };
};
const retain = t => { allocated.add(t.buffer); return t; };
const errors = [];
device.addEventListener('uncapturederror', event => errors.push(event.error.message));
const close = (actual, expected, label, abs = 2e-4, rel = 2e-5) => {
  let maxAbs = 0;
  for (let i = 0; i < expected.length; i++) {
    const error = Math.abs(actual[i] - expected[i]); maxAbs = Math.max(maxAbs, error);
    assert.ok(Number.isFinite(actual[i]) && error <= abs + rel*Math.abs(expected[i]), `${label}[${i}] ${actual[i]} vs ${expected[i]}`);
  }
  return maxAbs;
};
const sigmoid = x => 1/(1+Math.exp(-Math.max(-15, Math.min(15,x))));
try {
  for (const dtype of ['q4_0', 'q4_1', 'q5_0', 'q8_0', 'q6k']) {
    const blockValues = dtype === 'q6k' ? 256 : 32;
    for (const [M,N,K,offsetBank,basis] of [
      [1,1,blockValues,0,false], [1,7,blockValues*3,1,false], [3,9,blockValues*8,0,false],
      [blockValues,3,blockValues,0,true],
    ]) {
    const w=packed(dtype,offsetBank ? [2,N,K] : [N,K]);
    const values=Float32Array.from({length:(M+offsetBank)*K},(_,i)=>basis ? Number(Math.floor(i/K)===i%K) : Math.sin(i*0.23)*0.125);
    const input=tensor(values,[M+offsetBank,K]);
    const expected=projectReference(values.subarray(offsetBank*K),w.decoded.subarray(offsetBank*N*K),M,N,K);
    for (const mode of ['immediate','recorded']) {
      const recorder=mode==='recorded'?createCommandRecorder('native_cooperative'):null;
      if(recorder)recorders.push(recorder);
      const output=upload(new Float32Array(M*N+16).fill(-12345));
      const prefix=offsetBank*5;
      const packedBytes=w.weight.buffer.size;
      // Exact logical bank bytes, independent of pooled allocation capacity.
      const strideBytes=N*K/blockValues*({q4_0:18,q4_1:20,q5_0:22,q8_0:34,q6k:210}[dtype]);
      assert.ok(packedBytes >= strideBytes*(offsetBank+1));
      const options={outputDtype:'f32',variant:`${dtype}_cooperative`,outputBuffer:output,
        aOffset:offsetBank*K*4,bOffset:offsetBank*strideBytes,cOffset:prefix*4};
      const result=recorder?await recordQ40Matmul(recorder,input,w.weight,M,N,K,options):await runQ40Matmul(input,w.weight,M,N,K,options);
      if(recorder)await recorder.submitAndWait();
      const actual=new Float32Array(await readBuffer(result.buffer,(M*N+16)*4));
      const maxAbs=close(actual.subarray(prefix,prefix+M*N),expected,`${dtype}/${mode}`);
      assert.ok(actual.subarray(0,prefix).every(x=>x===-12345));
      assert.ok(actual.subarray(prefix+M*N).every(x=>x===-12345));
      console.log(JSON.stringify({dtype,M,N,K,offsetBank,basis,mode,maxAbs}));
    }
    }
  }
  for (const [routedDtype, downDtype, sharedDtype, T, sharedSize, limit] of [
    ['q4_0','q4_0','q8_0',1,32,null], ['q4_0','q4_1','q5_0',3,96,0.125], ['q8_0','q8_0','q4_0',2,64,null],
    ['q5_0','q5_0','q4_1',3,96,null], ['q4_1','q4_1','q5_0',2,32,null],
  ]) {
    const H=64,I=32,E=4,topK=3,stride=Math.max(I,sharedSize);
    const g=packed(routedDtype,[E,I,H]),u=packed(routedDtype,[E,I,H]),d=packed(downDtype,[E,H,I]);
    const sg=packed(sharedDtype,[sharedSize,H]),su=packed(sharedDtype,[sharedSize,H]),sd=packed(sharedDtype,[H,sharedSize]);
    const values=Float32Array.from({length:T*H},(_,i)=>Math.sin(i*0.31)*0.01),input=tensor(values,[T,H]);
    const idx=Uint32Array.from({length:T*topK},(_,i)=>(i*3+1)%E),rw=Float32Array.from({length:T*topK},(_,i)=>[0.5,0,0.25][i%topK]);
    const gates=Float32Array.from({length:T},(_,i)=>[-20,0,20][i%3]),gate=tensor(gates,[T,1]);
    const indices=upload(idx),routeWeights=upload(rw);
    const expectedH=new Float32Array(T*(topK+1)*stride),expected=new Float32Array(T*H);
    const clamp=x=>limit>0?Math.max(-limit,Math.min(limit,x)):x;
    for(let t=0;t<T;t++)for(let s=0;s<=topK;s++) {
      const shared=s===topK,dim=shared?sharedSize:I,e=shared?0:idx[t*topK+s];
      const gg=shared?sg.decoded:g.decoded,uu=shared?su.decoded:u.decoded,dd=shared?sd.decoded:d.decoded;
      for(let r=0;r<dim;r++) {
        let a=0,b=0;for(let k=0;k<H;k++){a+=values[t*H+k]*gg[(e*dim+r)*H+k];b+=values[t*H+k]*uu[(e*dim+r)*H+k];}
        expectedH[(t*(topK+1)+s)*stride+r]=clamp(a*sigmoid(a)*b);
      }
      for(let r=0;r<H;r++) {
        let sum=0;for(let k=0;k<dim;k++)sum+=expectedH[(t*(topK+1)+s)*stride+k]*dd[(e*H+r)*dim+k];
        expected[t*H+r]+=shared?clamp(sigmoid(gates[t])*sum):rw[t*topK+s]*sum;
      }
    }
    for(const mode of ['immediate','recorded']) {
      const recorder=mode==='recorded'?createCommandRecorder('native_shared_slot'):null;if(recorder)recorders.push(recorder);
      const options={numTokens:T,numExperts:E,topK,hiddenSize:H,intermediateSize:I,sharedSize,swigluLimit:limit,indices};
      const activated=retain(await executeSharedSlotGateUp(recorder,input,{gate:g.weight,up:u.weight,sharedGate:sg.weight,sharedUp:su.weight},options));
      const output=upload(new Float32Array(T*H+16).fill(-12345));
      const result=await executeSharedSlotDown(recorder,activated,{down:d.weight,sharedDown:sd.weight},gate,{...options,routeWeights,outputBuffer:output});
      const outer=retain(recorder?await recordResidualAdd(recorder,result,input,T*H):await runResidualAdd(result,input,T*H));
      if(recorder)await recorder.submitAndWait();
      const actualH=new Float32Array(await readBuffer(activated.buffer,expectedH.byteLength));
      for(let t=0;t<T;t++)for(let s=0;s<=topK;s++) {
        const base=(t*(topK+1)+s)*stride,dim=s===topK?sharedSize:I;
        close(actualH.subarray(base,base+dim),expectedH.subarray(base,base+dim),'shared GU');
      }
      const actual=new Float32Array(await readBuffer(output,(T*H+16)*4));
      const maxAbs=close(actual,expected,'shared down');assert.ok(actual.subarray(T*H).every(x=>x===-12345));
      const outerExpected=Float32Array.from(expected,(x,i)=>x+values[i]);
      close(new Float32Array(await readBuffer(outer.buffer,outerExpected.byteLength)),outerExpected,'exactly-once outer residual');
      console.log(JSON.stringify({routedDtype,downDtype,sharedDtype,T,sharedSize,limit,mode,maxAbs,outerResidual:'once'}));
    }
  }
  assert.deepEqual(errors,[]);
} finally {
  for(const recorder of recorders)await recorder.abort();
  await device.queue.onSubmittedWorkDone();
  for(const buffer of allocated)releaseBuffer(buffer);
}

}
