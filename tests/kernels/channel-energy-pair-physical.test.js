import assert from 'node:assert/strict';
import {probeNodeGPU} from '../helpers/gpu-probe.js';
import {getDevice, destroyDevice} from '../../src/gpu/device.js';
import {releaseNodeWebGPU} from '../../src/tooling/node-webgpu.js';
import {acquireBuffer, uploadData, readBuffer, releaseBuffer} from '../../src/memory/buffer-pool.js';
import {createTensor} from '../../src/gpu/tensor.js';
import {createCommandRecorder} from '../../src/gpu/command-recorder.js';
import {createChannelGainRequest} from "../../src/inference/pipelines/text/generator/activation/channel-gain.js";
import {deriveChannelGains} from '../../src/gpu/kernels/channel-gain.js';
const probe=await probeNodeGPU({installFileFetchShim:true});
assert.ok(probe.ready,probe.reason);
const device=getDevice(),owned=new Set();
const own=t=>{owned.add(t.buffer);return t;};
const values=async t=>[...new Float32Array(await readBuffer(t.buffer,8))];
let request,recorder;
device.pushErrorScope('validation');
try {
  const data=new Float32Array([100,100,1,-2,-3,4,5,-6,-7,8,100,100]);
  const buffer=acquireBuffer(data.byteLength);owned.add(buffer);uploadData(buffer,data);
  const input=createTensor(buffer,'f32',[6,2]);
  for(const recorded of [false,true]) {
    recorder=recorded?createCommandRecorder('same-pass-energy-pair'):null;
    request=createChannelGainRequest({mode:'capture',requestId:'pair',gainHash:'',cacheArm:'base-cache',
      tokenCount:6,span:{start:2,stop:4,templatePositions:[0,5]},captureContentSpan:{start:1,stop:5},
      layers:[{layerIdx:2,channels:2}]});
    await request.capture(2,input,6,2,recorder);
    if(recorder)await recorder.submitAndWait();
    const instruction=own(request.takeEnergies().get(2));
    const reference=own(request.takeReferenceEnergies().get(2));
    request.close();request=null;recorder=null;
    assert.deepEqual(await values(instruction),[17,26]);
    assert.deepEqual(await values(reference),[25,34]);
    const zero=own(await deriveChannelGains(null,instruction,reference,{channels:2,beta:0,clip:1,epsilon:1e-6}));
    assert.deepEqual(await values(zero),[1,1]);
    const positive=own(await deriveChannelGains(null,instruction,reference,{channels:2,beta:.05,clip:1,epsilon:1e-6}));
    const negative=own(await deriveChannelGains(null,instruction,reference,{channels:2,beta:-.05,clip:1,epsilon:1e-6}));
    const pos=await values(positive),neg=await values(negative);
    pos.forEach((v,i)=>assert.ok(Math.abs(v*neg[i]-1)<1e-6));
  }
  assert.equal(await device.popErrorScope(),null);
  console.log(JSON.stringify({transports:2,energyExact:true,zeroIdentity:true,reciprocalTolerance:1e-6,
    modelRequests:0,claim:'operator qualification only'}));
} finally {
  if(recorder)await recorder.submitAndWait();
  await device.queue.onSubmittedWorkDone();request?.close();
  for(const buffer of owned)releaseBuffer(buffer);
  destroyDevice();await releaseNodeWebGPU();
}
