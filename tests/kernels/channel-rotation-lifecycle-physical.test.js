import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, getKernelCapabilities, destroyDevice } from '../../src/gpu/device.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import { acquireBuffer, releaseBuffer, readBuffer, uploadData } from '../../src/memory/buffer-pool.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createCommandRecorder } from '../../src/gpu/command-recorder.js';
import { runDenseFFNGPU } from '../../src/inference/pipelines/text/ffn/dense.js';
import { createChannelGainRequest, freezeChannelRotations } from "../../src/inference/pipelines/text/generator/activation/channel-gain.js";
import { rotationReference, applyRotationReference } from './reference/channel-rotation.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
assert.ok(probe.ready, probe.reason);
const device = getDevice();
assert.doesNotMatch(JSON.stringify(getKernelCapabilities().adapterInfo), /swiftshader|llvmpipe|software/i);
const buffers = new Set();
const requests = new Set();
const handles = new Set();
const keep = tensor => { buffers.add(tensor.buffer); return tensor; };
const request = spec => { const value = createChannelGainRequest(spec); requests.add(value); return value; };
function tensor(values, shape) {
  const array = new Float32Array(values);
  const buffer = acquireBuffer(array.byteLength, undefined, 'rotation-lifecycle-fixture');
  buffers.add(buffer); uploadData(buffer, array); return createTensor(buffer, 'f32', shape);
}
const read = async tensor => new Float32Array(await readBuffer(tensor.buffer, tensor.shape.reduce((a,b) => a*b,1)*4));
function near(actual, expected) {
  actual.forEach((x,i) => assert.ok(Number.isFinite(x) && Math.abs(x-expected[i]) <= 3e-5*Math.max(1,Math.abs(expected[i])), `${i}: ${x} vs ${expected[i]}`));
}
let recorder = null;
try {
  const input = tensor([1,2,3,4], [2,2]);
  const identity = tensor([1,0,0,1], [2,2]);
  const weight = { buffer: identity.buffer, dtype:'f32', shape:[2,2], layout:'row' };
  const weights = { gate:weight, up:weight, down:weight };
  const context = { config:{ hiddenSize:2, intermediateSize:2, hiddenActivation:'silu', gatedActivation:true,
    swigluLimit:null, useDoubleWideMlp:false, useMoE:false }, phase:'prefill', currentSeqLen:0,
    activationDtype:'f32', kernelPath:null, recorder:null };
  const spec = { transformKind:'rotation', mode:'capture', requestId:'rotation-capture', gainHash:'',
    cacheArm:'base-cache', tokenCount:2, span:{ start:0, stop:1, templatePositions:[] },
    captureContentSpan:{ start:0, stop:2 }, rotationEpsilon:1e-6, layers:[{layerIdx:2,channels:2}] };
  const expectedActivation = [1,2,3,4].map(x => x*x/(1+Math.exp(-x)));
  const expectedPlane = rotationReference(expectedActivation,{tokens:2,channels:2,start:0,stop:1,contentStart:0,contentStop:2,epsilon:1e-6});
  for (const transport of ['immediate','recorded']) {
    const capture = request(spec);
    recorder = transport === 'recorded' ? createCommandRecorder('rotation-capture') : null;
    const capturedBase = keep(await runDenseFFNGPU(2,input,2,{...context,recorder,channelGainRequest:capture},weights));
    if(recorder) await recorder.submitAndWait(); recorder = null;
    near(await read(capturedBase),expectedActivation);
    const planes = capture.takeRotations(); const plane = planes.get(2);
    keep(plane.basis); keep(plane.status);
    await assert.rejects(capture.capture(2,input,2,2,null), /already captured/);
    capture.close();
    const frozen = await freezeChannelRotations([{layerIdx:2,channels:2,theta:0.05,rotation:plane}]); handles.add(frozen);
    const negative = await freezeChannelRotations([{layerIdx:2,channels:2,theta:-0.05,rotation:plane}]); handles.add(negative);
    const zero = await freezeChannelRotations([{layerIdx:2,channels:2,theta:0,rotation:plane}]); handles.add(zero);
    // Mutating public capture cannot change copied, opaque request state.
    uploadData(plane.basis.buffer, new Float32Array([0,0,0,0]));
    const applySpec = {...spec,mode:'apply',captureContentSpan:undefined,rotationEpsilon:undefined,
      gainHash:frozen.gainHash,frozenGains:frozen};
    const hybrid = request(applySpec);
    assert.throws(() => hybrid.validateLayer(2,context,2,2), /after the unmodified prefill/);
    assert.throws(() => frozen.close(), /active rotation/);
    assert.throws(() => createChannelGainRequest({...applySpec,transformKind:'gain'}), /kind/);
    assert.throws(() => createChannelGainRequest({...applySpec,gainHash:negative.gainHash}), /opaque/);
    for (const [handle,theta] of [[frozen,0.05],[negative,-0.05],[zero,0]]) {
      const applied = request({...applySpec,cacheArm:'re-prefill',gainHash:handle.gainHash,frozenGains:handle});
      recorder = transport === 'recorded' ? createCommandRecorder('rotation-apply') : null;
      const output = keep(await runDenseFFNGPU(2,input,2,{...context,recorder,channelGainRequest:applied},weights));
      if(recorder) await recorder.submitAndWait(); recorder = null;
      near(await read(output),applyRotationReference(expectedActivation,expectedPlane,{channels:2,theta}));
      assert.throws(() => applied.validateLayer(2,context,2,2), /repeated capture/);
      applied.close();
    }
    const decode = keep(await runDenseFFNGPU(2,input,1,{...context,phase:'decode',currentSeqLen:2,channelGainRequest:hybrid},weights));
    near(await read(decode),applyRotationReference(expectedActivation.slice(0,2),expectedPlane,{channels:2,theta:0.05}));
    hybrid.close(); frozen.close(); negative.close(); zero.close();
    assert.throws(() => createChannelGainRequest(applySpec), /opaque/);
    const unchanged = keep(await runDenseFFNGPU(2,input,2,context,weights));
    near(await read(unchanged),expectedActivation);
  }
  const aborted = request(spec);
  recorder = createCommandRecorder('rotation-abort');
  const discarded = keep(await runDenseFFNGPU(2,input,2,{...context,recorder,channelGainRequest:aborted},weights));
  assert.ok(discarded.buffer); recorder.abort(); recorder = null; aborted.close();
  assert.throws(() => aborted.has(2), /closed/);
  near(await read(keep(await runDenseFFNGPU(2,input,2,context,weights))),expectedActivation);
  console.log('rotation dense FFN lifecycle: immediate/recorded capture, zero and signed controls, private-copy isolation, cache arms, close and base restoration passed');
} finally {
  if(recorder) recorder.abort();
  for(const value of requests) value.close();
  for(const value of handles) value.close();
  await device.queue.onSubmittedWorkDone();
  for(const buffer of buffers) releaseBuffer(buffer);
  destroyDevice(); releaseNodeWebGPU();
}
