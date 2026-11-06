import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice, getKernelCapabilities } from '../../src/gpu/device.js';
import { acquireBuffer, uploadData, readBuffer, releaseBuffer } from '../../src/memory/buffer-pool.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import {
  runHyperConnectionBranch, recordHyperConnectionBranch,
  runHyperConnectionInject, recordHyperConnectionInject,
} from '../../src/gpu/kernels/hyper-connection.js';

// The source materializes mix*norm and branch*gate before summing. Fusing
// a*b-1 would produce -2^-46 here; rounding a*b first must produce exactly 0.
// Odd widths exercise incomplete workgroups and stream/token boundaries.
const probe = await probeNodeGPU({ installFileFetchShim: true });
assert(probe.ready, `Physical rounding check unavailable: ${probe.reason}`);
const adapter = getKernelCapabilities()?.adapterInfo;
assert(!/swiftshader|llvmpipe|lavapipe/i.test(JSON.stringify(adapter)));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const sourceSha256 = {};
for (const file of ['hyper_connection_branch.wgsl', 'hyper_connection_inject.wgsl']) {
  sourceSha256[file] = hash(await readFile(new URL(`../../src/gpu/kernels/${file}`, import.meta.url)));
}
const testSha256 = hash(await readFile(new URL(import.meta.url)));
const owned = new Set(), errors = [], results = [];
getDevice().addEventListener('uncapturederror', event => errors.push(event.error.message));
const upload = (data, shape) => {
  const buffer = acquireBuffer(data.byteLength);
  owned.add(buffer);
  uploadData(buffer, data);
  return createTensor(buffer, 'f32', shape);
};
const a = Math.fround(1 + 2 ** -23), b = Math.fround(1 - 2 ** -23);
assert.equal(Math.fround(a * b), 1);
assert.notEqual(Math.fround(a * b - 1), 0, 'fixture must distinguish fused arithmetic');
try {
  for (const width of [65, 257]) {
    const tokens = 3, streams = 4;
    const geometry = { numTokens: tokens, streamCount: streams, streamWidth: width };
    const mix = new Float32Array(tokens * streams * width), norm = new Float32Array(mix.length);
    for (let t = 0; t < tokens; t++) {
      for (let w = 0; w < width; w++) {
        mix[t * streams * width + w] = 1;
        norm[t * streams * width + w] = w % 2 === 0 ? -1 : 1;
        mix[(t * streams + 1) * width + w] = b;
        norm[(t * streams + 1) * width + w] = a;
      }
    }
    const residual = Float32Array.from({ length: norm.length }, (_, i) => i % width % 2 === 0 ? -1 : 1);
    const x = upload(residual, [tokens, streams * width]);
    const branch = upload(new Float32Array(tokens * width).fill(a), [tokens, width]);
    const gate = upload(new Float32Array(tokens * streams).fill(b), [tokens, streams]);
    const m = upload(mix, [tokens, streams * width]), n = upload(norm, [tokens, streams * width]);
    for (const mode of ['immediate', 'recorded']) {
      const recorder = mode === 'recorded' ? new CommandRecorder(getDevice(), 'hc_rounding') : null;
      try {
        const branched = recorder ? await recordHyperConnectionBranch(recorder, m, n, geometry)
          : await runHyperConnectionBranch(m, n, geometry);
        owned.add(branched.buffer);
        const injected = recorder ? await recordHyperConnectionInject(recorder, x, branch, gate, geometry)
          : await runHyperConnectionInject(x, branch, gate, geometry);
        owned.add(injected.buffer);
        if (recorder) await recorder.submitAndWait();
        for (const [operation, tensor, count] of [['branch', branched, tokens * width], ['inject', injected, norm.length]]) {
          const values = new Float32Array(await readBuffer(tensor.buffer, count * 4));
          const mismatches = Array.from(values).filter((value, i) => value !== (i % width % 2 === 0 ? 0 : operation === 'branch' ? 0.5 : 2)).length;
          results.push({ width, mode, operation, count, mismatches });
        }
      } finally {
        await recorder?.abort();
      }
    }
  }
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ test: 'hyper-connection-rounding-gpu', testSha256, sourceSha256, adapter, results }));
  assert(results.every(row => row.mismatches === 0), 'source requires rounded products before addition');
} finally {
  await getDevice()?.queue.onSubmittedWorkDone();
  for (const buffer of owned) releaseBuffer(buffer);
  destroyDevice();
  await releaseNodeWebGPU();
}
