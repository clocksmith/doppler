import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { packedFixture, decodePacked, projectReference } from '../helpers/q4_0-fixture.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { createCommandRecorder } from '../../src/gpu/command-recorder.js';
import { PreparedDecodeWorkspace } from '../../src/gpu/prepared-decode-workspace.js';
import { recordRMSNorm } from '../../src/gpu/kernels/rmsnorm.js';
import { recordMatmul } from '../../src/gpu/kernels/matmul.js';
import { recordQ40Matmul } from '../../src/gpu/kernels/q4_0.js';
import { acquireBuffer, releaseBuffer, uploadData, readBuffer } from '../../src/memory/buffer-pool.js';

assert.ok((await probeNodeGPU({ installFileFetchShim: true })).ready);
const device = getDevice(), workspace = new PreparedDecodeWorkspace(device), owned = [], errors = [];
device.addEventListener('uncapturederror', event => errors.push(event.error.message));
const upload = data => { const b = acquireBuffer(data.byteLength); owned.push(b); uploadData(b, data); return b; };
const M_VALUES = [1, 4, 7], N = 19, K = 512;
const scales = Float32Array.from({ length: K }, (_, i) => 0.5 + (i % 7) * 0.125);
const normWeight = createWeightBuffer(upload(scales), 'f32', 'row', [K]);
const matrix = Float32Array.from({ length: N * K }, (_, i) => Math.sin(i * 0.003) * 0.1);
const denseWeight = createWeightBuffer(upload(matrix), 'f32', 'row', [N, K]);
const step = { op: 'q_proj', kernel: 'matmul_f32_row_subgroup.wgsl', entry: 'main', constants: { WORKGROUP_SIZE: 128, COLS_PER_WG: 4 } };
const kernelPath = { activationDtype: 'f32', decode: { steps: [step] }, prefill: { steps: [step] } };
const cases = [];
try {
  for (const M of M_VALUES) {
    const data = new Float32Array(M * K), buffer = upload(data);
    const first = new Map();
    for (let iteration = 0; iteration < 2; iteration++) {
      for (let i = 0; i < data.length; i++) data[i] = Math.cos(i * 0.07 + iteration * 0.3) * 0.2;
      uploadData(buffer, data);
      const normalized = new Float32Array(data.length);
      for (let row = 0; row < M; row++) {
        let sum = 0; for (let k = 0; k < K; k++) sum += data[row * K + k] ** 2;
        for (let k = 0; k < K; k++) normalized[row * K + k] = data[row * K + k] * scales[k] / Math.sqrt(sum / K + 1e-6);
      }
      const rec = createCommandRecorder('prepared_batch_projection'); rec.decodeWorkspace = workspace;
      const norm = await recordRMSNorm(rec, createTensor(buffer, 'f32', [M, K]), normWeight, 1e-6,
        { batchSize: M, hiddenSize: K, layerIdx: 0, role: 'input_norm' });
      assert.ok(workspace.ownsBuffer(norm.buffer));
      const results = new Map();
      const dense = await recordMatmul(rec, norm, denseWeight, M, N, K,
        { role: 'q_proj', layerIdx: 0, outputDtype: 'f32', transposeB: true, kernelPath });
      results.set('f32', { result: dense, expected: projectReference(normalized, matrix, M, N, K) });
      for (const [dtype, variant] of [['q4_0', 'q4_0_pair_columns'], ['q8_0', 'q8_0_pair_columns'], ['q6k', 'q6k_pair_columns']]) {
        const packed = packedFixture(dtype, N, K), decoded = decodePacked(dtype, packed, N, K);
        const weight = createWeightBuffer(upload(packed), dtype, 'row', [N, K]);
        const result = await recordQ40Matmul(rec, norm, weight, M, N, K,
          { variant, outputDtype: 'f32', role: dtype, layerIdx: 0 });
        assert.equal(workspace.ownsBuffer(result.buffer), M > 1, 'single-token GGUF allocation must stay unchanged');
        if (!workspace.ownsBuffer(result.buffer)) owned.push(result.buffer);
        results.set(dtype, { result, expected: projectReference(normalized, decoded, M, N, K) });
      }
      for (const { result } of results.values()) if (workspace.ownsBuffer(result.buffer)) rec.trackTemporaryBuffer(result.buffer);
      rec.trackTemporaryBuffer(norm.buffer); await rec.submitAndWait();
      for (const [dtype, { result, expected }] of results) {
        if (iteration === 0) first.set(dtype, result.buffer);
        else if (dtype === 'f32' || M > 1) assert.equal(result.buffer, first.get(dtype), 'stable batch slot changed across recorders');
        const values = new Float32Array(await readBuffer(result.buffer, M * N * 4));
        for (let i = 0; i < values.length; i++) assert.ok(Number.isFinite(values[i])
          && Math.abs(values[i] - expected[i]) <= 2e-4 + 3e-5 * Math.abs(expected[i]), `${dtype}/${M}/${iteration}/${i}`);
      }
      cases.push({ M, iteration, workspace: workspace.getStats() });
    }
  }
  assert.ok(workspace.getStats().hits > 0, 'no stable binding was reused');
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ passed: true, scope: 'session-owned row-batch projections and normalization', cases }));
} finally {
  await device.queue.onSubmittedWorkDone(); workspace.release(); for (const b of owned) releaseBuffer(b); destroyDevice();
}
