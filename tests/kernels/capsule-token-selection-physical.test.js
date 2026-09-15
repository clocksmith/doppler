import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, getKernelCapabilities, destroyDevice } from '../../src/gpu/device.js';
import { sampleCapsuleLogits } from '../../src/client/runtime/session-controller.js';
import { selectTokenFromGpuLogits } from '../../src/inference/pipelines/text/generator/token-selection.js';
import { clearPipelineCaches } from '../../src/gpu/kernels/pipeline-cache.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) throw new Error(`Physical token-selection test requires WebGPU: ${probe.reason}`);
const device = getDevice();
const adapter = getKernelCapabilities().adapterInfo;
assert.doesNotMatch(JSON.stringify(adapter), /swiftshader|llvmpipe|software/i);
const base = { temperature: 0, topK: 1, topP: 1, seed: 7, repetitionPenalty: 1,
  repetitionPenaltyWindow: 0, presencePenalty: 0, suppressTokenIds: [] };
let cases = 0;
const copies = [];
const createEncoder = device.createCommandEncoder.bind(device);
device.createCommandEncoder = options => {
  const encoder = createEncoder(options), copy = encoder.copyBufferToBuffer.bind(encoder);
  encoder.copyBufferToBuffer = (source, offset, destination, destinationOffset, size) => {
    if (destination.usage & GPUBufferUsage.MAP_READ) copies.push(size);
    return copy(source, offset, destination, destinationOffset, size);
  };
  return encoder;
};
const upload = values => {
  const buffer = device.createBuffer({ size: values.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  device.queue.writeBuffer(buffer, 0, values);
  return { logitsBuffer: buffer, logitsDtype: 'f32', vocabSize: values.length };
};
async function compare(values, history, options, tokenContract = { padTokenId: null }) {
  const result = upload(values);
  const expected = sampleCapsuleLogits(values, history, { ...base, ...options }, tokenContract);
  try {
    const observed = await selectTokenFromGpuLogits(result, history, { ...base, ...options }, tokenContract);
    assert.equal(observed.tokenId, expected, JSON.stringify({ size: values.length, history, options }));
    cases++;
  } finally { result.logitsBuffer.destroy(); }
}
try {
  for (const size of [5, 257, 4096, 151936]) {
    const values = new Float32Array(size).fill(-100);
    values.set([8, -4, 6, 3, 8]);
    for (const temperature of [0, 0.001, 1]) {
      for (const topK of [0, 1, 4]) {
        for (const topP of [0.1, 0.8, 1]) {
          await compare(values, [0, 0, 4, 2], { temperature, topK, topP,
            repetitionPenalty: 2, presencePenalty: 0.5, suppressTokenIds: [3, 3, size + 1] });
        }
      }
    }
  }
  for (const window of [0, 1, 3, 10]) {
    for (const seed of [0, 1, 3, 7, 42]) {
      await compare(Float32Array.from([8, -4, 6, 3, 8]), [0, 0, 4, 2], {
        repetitionPenaltyWindow: window, repetitionPenalty: 1.25, presencePenalty: 0.5,
        temperature: 0.8, topK: 0, topP: 0.95, seed,
      }, { padTokenId: 0 });
    }
  }
  await compare(Float32Array.from([NaN, Infinity, -Infinity, 2, 2]), [], {}, { padTokenId: 3 });
  for (const values of [Float32Array.from([NaN, Infinity, -Infinity]), Float32Array.from([1, 2])]) {
    const result = upload(values);
    try {
      await assert.rejects(selectTokenFromGpuLogits(result, [], { ...base, suppressTokenIds: [0, 1] }, { padTokenId: null }), /finite candidate/);
    } finally { result.logitsBuffer.destroy(); }
  }
  // Delayed real compilation: cancellation must prevent queue submission.
  clearPipelineCaches();
  const controller = new AbortController();
  const result = upload(Float32Array.from([1, 2, 3]));
  const compile = device.createComputePipelineAsync.bind(device);
  const submit = device.queue.submit.bind(device.queue);
  let submissions = 0;
  device.createComputePipelineAsync = async descriptor => { controller.abort(); return compile(descriptor); };
  device.queue.submit = commands => { submissions++; return submit(commands); };
  try {
    await assert.rejects(selectTokenFromGpuLogits(result, [], { ...base, signal: controller.signal }, { padTokenId: null }), /aborted/);
    assert.equal(submissions, 0);
  } finally {
    device.createComputePipelineAsync = compile; device.queue.submit = submit;
    result.logitsBuffer.destroy();
  }
  assert(copies.length > cases);
  assert(copies.every(size => size === 4), 'selection reads back only a token, including failure sentinel');
  console.log(JSON.stringify({ test: 'capsule-token-selection-physical', cases, readbacks: copies.length,
    readbackBytesPerSelection: 4, adapter, scope: 'Operator and cancellation checks; not model or performance qualification.' }));
} finally {
  device.createCommandEncoder = createEncoder;
  destroyDevice();
}
