import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, getKernelCapabilities, destroyDevice } from '../../src/gpu/device.js';
import { createCommandRecorder } from '../../src/gpu/command-recorder.js';
import { runGPUSample, recordGPUSample } from '../../src/gpu/kernels/sample.js';
import { readBuffer, releaseBuffer } from '../../src/memory/buffer-pool.js';
import { sample } from '../../src/inference/token-sampling.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) throw new Error(`Physical sampling test requires WebGPU: ${probe.reason}`);
const device = getDevice();
const caps = getKernelCapabilities();
assert.doesNotMatch(JSON.stringify(caps.adapterInfo), /swiftshader|llvmpipe|software/i);
let cases = 0;
try {
  // The former partition-max approximation discarded the second winner.
  for (const vocabSize of [4, 512, 4096]) {
    const values = new Float32Array(vocabSize).fill(-100);
    values.set([3, 2, 1, 0]);
    const half = new Uint16Array(vocabSize).fill(0xd640);
    half.set([0x4200, 0x4000, 0x3c00, 0]);
    for (const logitsDtype of ['f32', 'f16']) {
      const bytes = logitsDtype === 'f32' ? values : half;
      const logits = device.createBuffer({ size: bytes.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(logits, 0, bytes);
      try {
        for (const topK of [0, 1, 2, 129, 8192]) {
          for (const topP of [0.1, 0.75, 1]) {
            for (const seed of [1, 3, 7]) {
              const expected = sample(values.slice(), { temperature: 1, topK, topP, seed, padTokenId: 3 });
              const options = { temperature: 1, topK, topP, randomSeed: seed, logitsDtype,
                padTokenId: 3, logitSoftcap: 0, outputIndex: 1, greedyThreshold: 0.01 };
              assert.equal(await runGPUSample(logits, vocabSize, options), expected,
                JSON.stringify({ vocabSize, logitsDtype, topK, topP, seed, adapter: 'immediate' }));
              const recorder = createCommandRecorder('sampling-distribution-parity');
              let output;
              try {
                output = await recordGPUSample(recorder, logits, vocabSize, options);
                await recorder.submitAndWait();
                assert.equal(new Uint32Array(await readBuffer(output))[1], expected,
                  JSON.stringify({ vocabSize, logitsDtype, topK, topP, seed, adapter: 'recorded' }));
              } finally { recorder.abort(); if (output) releaseBuffer(output); }
              cases += 2;
            }
          }
        }
      } finally { logits.destroy(); }
    }
  }
  console.log(JSON.stringify({ test: 'sampling-distribution-physical', cases, adapter: caps.adapterInfo,
    evidence: 'operator parity only; not application/model qualification or a performance comparison' }));
} finally { destroyDevice(); }
