import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { acquireBuffer, uploadData, readBuffer, releaseBuffer } from '../../src/memory/buffer-pool.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { runPleRowDecode, recordPleRowDecode } from '../../src/gpu/kernels/ple.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
assert.ok(probe.ready, probe.reason);
const owned = [];
try {
  // 160 values = five 18-byte blocks: row two starts at byte 90.
  // One row also exercises a two-byte padded tail; 3 rows cross a workgroup.
  for (const numTokens of [1, 2, 3]) {
    const embedDim = 160;
    const count = numTokens * embedDim;
    const bytes = new Uint8Array(Math.ceil(count / 32 * 18 / 4) * 4);
    const view = new DataView(bytes.buffer);
    const expected = new Float32Array(count);
    for (let block = 0; block < count / 32; block++) {
      const negative = block % 2 === 1;
      const scaleBits = negative ? 0xb800 : 0x3c00; // -0.5 or +1
      const scale = negative ? -0.5 : 1;
      view.setUint16(block * 18, scaleBits, true);
      for (let lane = 0; lane < 16; lane++) {
        const low = (lane + block) % 16;
        const high = (15 - lane + block) % 16;
        bytes[block * 18 + 2 + lane] = low | (high << 4);
        expected[block * 32 + lane] = scale * (low - 8);
        expected[block * 32 + 16 + lane] = scale * (high - 8);
      }
    }
    const rows = acquireBuffer(bytes.byteLength, undefined, 'q4_0_rows');
    owned.push(rows);
    uploadData(rows, bytes);
    for (const recorded of [false, true]) {
      const options = { numTokens, embedDim, tableDtype: 'q4_0' };
      const recorder = recorded ? new CommandRecorder(getDevice(), 'q4_0_rows') : null;
      const tensor = recorder
        ? await recordPleRowDecode(recorder, rows, options)
        : await runPleRowDecode(rows, options);
      owned.push(tensor.buffer);
      if (recorder) await recorder.submitAndWait();
      assert.deepEqual(new Float32Array(await readBuffer(tensor.buffer, count * 4)), expected);
    }
  }
  const bf16 = acquireBuffer(8, undefined, 'bf16_rows');
  owned.push(bf16);
  uploadData(bf16, Uint16Array.of(0x3f80, 0xc000, 0x0000, 0x3f00));
  const decoded = await runPleRowDecode(bf16, { numTokens: 1, embedDim: 4, tableDtype: 'bf16' });
  owned.push(decoded.buffer);
  assert.deepEqual([...new Float32Array(await readBuffer(decoded.buffer, 16))], [1, -2, 0, 0.5]);
  await assert.rejects(runPleRowDecode({ size: 16 }, { numTokens: 1, embedDim: 32, tableDtype: 'q4_0' }), /too small/);
  await assert.rejects(runPleRowDecode(bf16, { numTokens: 1, embedDim: 34, tableDtype: 'q4_0' }), /align/);
  await assert.rejects(runPleRowDecode(bf16, { numTokens: 1, embedDim: 4 }), /tableDtype/);
  console.log('PLE BF16/Q4_0 row decode: immediate/recorded GPU checks passed.');
} finally {
  for (const buffer of owned) releaseBuffer(buffer);
  destroyDevice();
}
