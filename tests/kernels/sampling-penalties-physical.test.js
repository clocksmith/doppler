import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, getKernelCapabilities, destroyDevice } from '../../src/gpu/device.js';
import { createCommandRecorder } from '../../src/gpu/command-recorder.js';
import { recordRepPenalty } from '../../src/gpu/kernels/rep-penalty.js';
import { readBuffer } from '../../src/memory/buffer-pool.js';
import { applyPresencePenalty, applyRepetitionPenalty } from '../../src/inference/pipelines/text/sampling.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) throw new Error(`Physical sampling test requires WebGPU: ${probe.reason}`);
const device = getDevice();
const caps = getKernelCapabilities();
assert.doesNotMatch(JSON.stringify(caps.adapterInfo), /swiftshader|llvmpipe|software/i);
let cases = 0;
try {
  for (const history of [[0, 0, 1, 2], [2, 0, 2, 0], []]) {
    for (const window of [0, 1, 3, 8]) {
      for (const presencePenalty of [0, 2]) {
        const batch = [3, 0, 0, 1]; // Leading element is outside the declared offset.
        const source = new Float32Array([8, -4, 6, 3]);
        const expected = source.slice();
        const context = [...history, ...batch.slice(1)];
        applyRepetitionPenalty(expected, context, 2, window);
        applyPresencePenalty(expected, context, presencePenalty, window);
        const buffers = [];
        const upload = data => {
          const buffer = device.createBuffer({ size: Math.max(4, data.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
          buffers.push(buffer);
          if (data.byteLength) device.queue.writeBuffer(buffer, 0, data);
          return buffer;
        };
        const recorder = createCommandRecorder('penalty-window-parity');
        try {
          const logits = upload(source);
          await recordRepPenalty(recorder, logits, upload(Uint32Array.from(history)), upload(Uint32Array.from(batch)), {
            vocabSize: source.length, historyCount: history.length, penalty: 2, presencePenalty,
            repetitionPenaltyWindow: window, batchCount: 3, batchOffset: 1, logitsDtype: 'f32',
          });
          await recorder.submitAndWait();
          assert.deepEqual(new Float32Array(await readBuffer(logits)), expected);
          cases++;
        } finally { recorder.abort(); for (const buffer of buffers) buffer.destroy(); }
      }
    }
  }
  console.log(JSON.stringify({ test: 'sampling-penalties-physical', cases, adapter: caps.adapterInfo,
    evidence: 'operator parity only; not application or model qualification' }));
} finally { destroyDevice(); }
