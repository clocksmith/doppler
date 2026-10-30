import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { acquireBuffer, uploadData, readBuffer, releaseBuffer } from '../../src/memory/buffer-pool.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { destroyDevice } from '../../src/gpu/device.js';
import { runGemma4Expert } from '../../src/inference/pipelines/text/moe/gpu-executor.js';
import { runQwenSharedExpert } from '../../src/inference/pipelines/text/ffn/qwen-shared-expert.js';
const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) throw new Error(`Physical Qwen shared-expert check unavailable: ${probe.reason}`);
const H = 4, I = 3, T = 2;
const input = [0.25, -0.5, 0.75, 1, -0.75, 0.1, 0.2, 0.5];
const routed = [0.1, 0.2, 0.3, 0.4, -0.4, -0.3, -0.2, -0.1];
const gate = [0.1,0.2,0.3,0.4,-0.2,0.4,0.1,0.3,0.5,-0.2,0.4,0.1];
const up = [-0.2,0.1,0.4,0.3,0.2,0.3,-0.1,0.2,0.5,0.1,0.2,-0.4];
const down = [0.1,0.2,0.3,-0.2,0.4,0.1,0.3,-0.2,0.5,0.1,0.2,-0.1];
const router = [0.5,-0.3,0.2,-0.7];
const allocated = [];
const buffer = data => { const b = acquireBuffer(data.length * 4); allocated.push(b); uploadData(b, new Float32Array(data)); return b; };
const matrix = (data, shape) => createWeightBuffer(buffer(data), 'f32', 'row', shape);
let output = null;
try {
  const w = { sharedExpertGate: matrix(gate,[I,H]), sharedExpertUp: matrix(up,[I,H]),
    sharedExpertDown: matrix(down,[H,I]), sharedExpertRouter: matrix(router,[1,H]) };
  output = await runQwenSharedExpert(createTensor(buffer(input),'f32',[T,H]),
    createTensor(buffer(routed),'f32',[T,H]),w,
    {numTokens:T,hiddenSize:H,intermediateSize:I,kernelPath:null,swigluLimit:null});
  const actual = new Float32Array(await readBuffer(output.buffer,T*H*4));
  const expected = [];
  const dot = (a,b) => a.reduce((s,v,i)=>s+v*b[i],0);
  for(let t=0;t<T;t++) {
    const x=input.slice(t*H,(t+1)*H);
    const activation=Array.from({length:I},(_,i)=> {
      const g=dot(x,gate.slice(i*H,(i+1)*H));
      return g/(1+Math.exp(-g))*dot(x,up.slice(i*H,(i+1)*H));
    });
    const scale=1/(1+Math.exp(-dot(x,router)));
    for(let h=0;h<H;h++) expected.push(routed[t*H+h]+scale*dot(activation,down.slice(h*I,(h+1)*I)));
  }
  for(let i=0;i<expected.length;i++) assert.ok(Number.isFinite(actual[i]) && Math.abs(actual[i]-expected[i])<0.0001,
    `shared expert[${i}]: ${actual[i]} versus ${expected[i]}`);
  // The 35B shared expert broadcasts 53 scalar gates across 2048 hidden values.
  // The old channel-axis dispatch exceeded the device's 65535-workgroup limit.
  const largeT = 53, largeH = 2048;
  const largeInput = Float32Array.from({length:largeT*largeH}, (_,i)=>Math.sin(i*0.13)*0.01);
  const largeRouted = Float32Array.from({length:largeT*largeH}, (_,i)=>Math.cos(i*0.17)*0.01);
  const largeGate = Float32Array.from({length:I*largeH}, (_,i)=>Math.cos(i*0.07)*0.01);
  const largeUp = Float32Array.from({length:I*largeH}, (_,i)=>Math.sin(i*0.11)*0.01);
  const largeDown = Float32Array.from({length:largeH*I}, (_,i)=>Math.cos(i*0.03)*0.01);
  const largeRouter = Float32Array.from({length:largeH}, (_,i)=>Math.sin(i*0.09)*0.01);
  const largeWeights = { sharedExpertGate:matrix(largeGate,[I,largeH]), sharedExpertUp:matrix(largeUp,[I,largeH]),
    sharedExpertDown:matrix(largeDown,[largeH,I]), sharedExpertRouter:matrix(largeRouter,[1,largeH]) };
  const largeOutput = await runQwenSharedExpert(createTensor(buffer(largeInput),'f32',[largeT,largeH]),
    createTensor(buffer(largeRouted),'f32',[largeT,largeH]),largeWeights,
    {numTokens:largeT,hiddenSize:largeH,intermediateSize:I,kernelPath:null,swigluLimit:null});
  let largeMaxAbs = 0;
  try {
    const observed = new Float32Array(await readBuffer(largeOutput.buffer,largeT*largeH*4));
    for(let t=0;t<largeT;t++) {
      const x=largeInput.subarray(t*largeH,(t+1)*largeH);
      const activation=Array.from({length:I},(_,i)=> {
        const g=dot(x,largeGate.subarray(i*largeH,(i+1)*largeH));
        return g/(1+Math.exp(-g))*dot(x,largeUp.subarray(i*largeH,(i+1)*largeH));
      });
      const scale=1/(1+Math.exp(-dot(x,largeRouter)));
      for(let h=0;h<largeH;h++) {
        const value=largeRouted[t*largeH+h]+scale*dot(activation,largeDown.subarray(h*I,(h+1)*I));
        const error=Math.abs(observed[t*largeH+h]-value);
        assert.ok(Number.isFinite(observed[t*largeH+h]) && error<0.0001,`shared 53x2048[${t},${h}]`);
        largeMaxAbs=Math.max(largeMaxAbs,error);
      }
    }
  } finally { releaseBuffer(largeOutput.buffer); }
  // Packed Qwen experts share buffers; nonzero expert and gathered-token
  // offsets must select the right bank and token, not expert zero's row.
  const packedHidden = 64;
  const packedInput = Array.from({length: T * packedHidden}, (_, i) =>
    i % packedHidden < H ? input[Math.floor(i / packedHidden) * H + i % packedHidden] : 0);
  const packedGate = Array.from({length: I * packedHidden}, (_, i) =>
    i % packedHidden < H ? gate[Math.floor(i / packedHidden) * H + i % packedHidden] : 0);
  const packedUp = Array.from({length: I * packedHidden}, (_, i) =>
    i % packedHidden < H ? up[Math.floor(i / packedHidden) * H + i % packedHidden] : 0);
  const packedDown = Array.from({length: packedHidden * I}, (_, i) => down[i % (H * I)]);
  // These nonzero offsets meet the production storage-binding alignment; the
  // runtime's 256-byte validation and numerical tolerance stay unchanged.
  const tokenOffset = packedHidden * 4;
  const gateUpOffset = 2 * I * packedHidden * 4;
  const downOffset = packedHidden * I * 4;
  for (const offset of [tokenOffset, gateUpOffset, downOffset]) assert.equal(offset % 256, 0);
  const expertOutput = buffer(new Array(T * packedHidden).fill(0));
  const packed = { numExperts: 2, expertIdx: 1, expertFormat: 'qwen3_5',
    gateUp: matrix([...new Array(2 * I * packedHidden).fill(0), ...packedGate, ...packedUp], [2, 2 * I, packedHidden]),
    down: matrix([...new Array(packedHidden * I).fill(0), ...packedDown], [2, packedHidden, I]) };
  await runGemma4Expert(createTensor(buffer(packedInput), 'f32', [T, packedHidden]), expertOutput, packed,
    1, tokenOffset, tokenOffset, packedHidden, I, 'f32', null, null, 'silu', {gateUp:'ffn_gate',down:'ffn_down'});
  const expertValues = new Float32Array(await readBuffer(expertOutput, T * packedHidden * 4));
  const x = packedInput.slice(packedHidden, 2 * packedHidden);
  const act = Array.from({length:I}, (_, i) => {
    const g = dot(x, packedGate.slice(i * packedHidden, (i + 1) * packedHidden));
    return g / (1 + Math.exp(-g)) * dot(x, packedUp.slice(i * packedHidden, (i + 1) * packedHidden));
  });
  for (let h = 0; h < packedHidden; h++) assert.ok(
    Number.isFinite(expertValues[packedHidden + h]) &&
    Math.abs(expertValues[packedHidden + h] - dot(act, packedDown.slice(h * I, (h + 1) * I))) < 0.0001,
    `packed expert nonzero-offset mismatch at ${h}`);
  console.log(JSON.stringify({test:'qwen35-shared-expert',passed:true,maxAbs:Math.max(...actual.map((v,i)=>Math.abs(v-expected[i]))),
    large:{numTokens:largeT,hiddenSize:largeH,maxAbs:largeMaxAbs},provider:probe.providerReceipt}));
} finally {
  if(output) releaseBuffer(output.buffer);
  for(const b of allocated) releaseBuffer(b);
  destroyDevice();
}
