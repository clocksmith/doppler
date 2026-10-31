import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice, getKernelCapabilities } from '../../src/gpu/device.js';
import { acquireBuffer, uploadData, readBuffer, releaseBuffer } from '../../src/memory/buffer-pool.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { runRepeatChannels, recordRepeatChannels } from '../../src/gpu/kernels/repeat_channels.js';
import { runSiLU, recordSiLU } from '../../src/gpu/kernels/silu.js';
import { runResidualAdd, recordResidualAdd } from '../../src/gpu/kernels/residual.js';
import { runSigmoidGatedAdd, recordSigmoidGatedAdd } from '../../src/gpu/kernels/sigmoid-gated-add.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) throw new Error(`Physical sigmoid gated add qualification unavailable: ${probe.reason}`);
const capabilities = getKernelCapabilities();
const adapter = JSON.stringify(capabilities?.adapterInfo ?? {}).toLowerCase();
assert.ok(/apple|m3/.test(adapter) && !/swiftshader|llvmpipe|lavapipe/.test(adapter), 'mac-host physical qualification requires observed Apple hardware');
console.log(JSON.stringify({ timeUTC: new Date().toISOString(), providerReceipt: probe.providerReceipt, capabilities, scope: 'Physical operator correctness; no timing claim' }));
const cases = [];
try {
  for (const [numTokens, hiddenSize, swigluLimit] of [
    [1, 1, null], [1, 2048, null], [32, 7, 0.125], [33, 2048, null], [53, 2048, 0], [53, 7, -1],
  ]) {
    const count = numTokens * hiddenSize;
    const values = Float32Array.from({ length: count }, (_, i) => Math.sin(i * 0.17) * 4);
    const residualValues = Float32Array.from({ length: count }, (_, i) => Math.cos(i * 0.13));
    const gates = Float32Array.from({ length: numTokens }, (_, i) => [-20, -15, -1, 0, 1, 15, 20][i % 7]);
    const owned = [];
    const recorders = [];
    const tensor = (data, shape) => {
      const buffer = acquireBuffer(data.byteLength);
      owned.push(buffer);
      uploadData(buffer, data);
      return createTensor(buffer, 'f32', shape);
    };
    const input = tensor(values, [numTokens, hiddenSize]);
    const gate = tensor(gates, [numTokens, 1]);
    const residual = tensor(residualValues, [numTokens, hiddenSize]);
    try {
      for (const mode of ['immediate', 'recorded']) {
        const recorder = mode === 'recorded' ? new CommandRecorder(getDevice(), 'sigmoid_gated_add_equivalence') : null;
        if (recorder) recorders.push(recorder);
        const repeat = recorder ? recordRepeatChannels.bind(null, recorder) : runRepeatChannels;
        const activate = recorder ? recordSiLU.bind(null, recorder) : runSiLU;
        const add = recorder ? recordResidualAdd.bind(null, recorder) : runResidualAdd;
        const combine = recorder ? recordSigmoidGatedAdd.bind(null, recorder) : runSigmoidGatedAdd;
        const expanded = await repeat(gate, { inChannels: numTokens, height: 1, width: 1, repeats: hiddenSize });
        const gated = await activate(input, { size: count, gate: expanded,
          inputActivation: 'identity', gateActivation: 'sigmoid', swigluLimit });
        const separate = await add(residual, gated, count);
        const combined = await combine(input, gate, residual, { numTokens, hiddenSize, swigluLimit });
        owned.push(expanded.buffer, gated.buffer, separate.buffer, combined.buffer);
        if (recorder) await recorder.submitAndWait();
        const expected = new Float32Array(await readBuffer(separate.buffer, count * 4));
        const actual = new Float32Array(await readBuffer(combined.buffer, count * 4));
        let maxAbs = 0;
        let different = 0;
        for (let i = 0; i < count; i++) {
          assert.ok(Number.isFinite(actual[i]));
          maxAbs = Math.max(maxAbs, Math.abs(actual[i] - expected[i]));
          if (actual[i] !== expected[i]) different++;
        }
        assert.ok(maxAbs <= 0.000001, `${mode}/${numTokens}/${hiddenSize}: maxAbs=${maxAbs}`);
        const result = { mode, numTokens, hiddenSize, swigluLimit, maxAbs, different,
          scope: 'Physical operator equivalence only; not full-model or performance acceptance' };
        cases.push(result);
        console.log(JSON.stringify(result));
      }
    } finally {
      for (const recorder of recorders) await recorder.abort();
      await getDevice().queue.onSubmittedWorkDone();
      for (const buffer of new Set(owned)) releaseBuffer(buffer);
    }
  }
  assert.equal(cases.length, 12);
} finally {
  await getDevice()?.queue.onSubmittedWorkDone();
  destroyDevice();
}
