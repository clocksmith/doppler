import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, getKernelCapabilities, destroyDevice } from '../../src/gpu/device.js';
import { createCommandRecorder } from '../../src/gpu/command-recorder.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { acquireBuffer, releaseBuffer, readBuffer, uploadData } from '../../src/memory/buffer-pool.js';
import { float32ToFloat16, float16ToFloat32 } from '../../src/formats/float16.js';
import { runSiLU, recordSiLU, runSiLURowSplit, recordSiLURowSplit } from '../../src/gpu/kernels/silu.js';

// SiLU/sigmoid are unbounded mathematical activations. Only an explicit
// swigluLimit may clip their combined output; overflow protection must not
// replace their input by +/-15. Binary64 references are independent of WGSL.
const outputPath = process.argv.slice(2).find(value => !value.startsWith('--'));
const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) {
  assert(!process.argv.includes('--require-gpu'), probe.reason);
  console.log(`silu-unclamped-physical: skipped (${probe.reason})`);
} else {
  const device = getDevice(), errors = [];
  device.addEventListener('uncapturederror', event => errors.push(event.error.message));
  const hash = value => createHash('sha256').update(value).digest('hex');
  const receipt = { schema: 'doppler.silu-unclamped-physical/v1',
    scope: 'Activation functions and explicit SwiGLU output clipping; not full-model qualification',
    adapter: getKernelCapabilities().adapterInfo, shaderSha256: {}, rows: [], errors };
  assert.doesNotMatch(JSON.stringify(receipt.adapter), /software|swiftshader|llvmpipe/i);
  const base = [-80, -40, -32, -20, -16, -15.01, -15, -14.99, -8, -4, -1, -0,
    0, 0.0001, 0.01, 1, 4, 8, 14.99, 15, 15.01, 16, 20, 32, 40, 80];
  const sigmoid = x => x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
  try {
    for (const dtype of ['f32', 'f16']) {
      const file = dtype === 'f32' ? 'silu.wgsl' : 'silu_f16.wgsl';
      receipt.shaderSha256[file] = hash(await readFile(new URL(`../../src/gpu/kernels/${file}`, import.meta.url)));
      const round = dtype === 'f32' ? Math.fround : value => float16ToFloat32(float32ToFloat16(value));
      const values = Array.from(base, round);
      for (const kind of ['unary', 'unary-vec4', 'sigmoid-gate', 'rowsplit', 'rowsplit-limited']) {
        const split = kind.startsWith('rowsplit'), limit = kind === 'rowsplit-limited' ? 0.125 : null;
        const numTokens = 2, dim = values.length;
        const inputValues = split
          ? [...values, ...values.map(() => 1.25), ...values.toReversed(), ...values.map(() => 1.25)]
          : kind === 'sigmoid-gate' ? values.map(() => 1) : values;
        const references = (split ? [...values, ...values.toReversed()] : values).map(value => {
          let result = kind === 'sigmoid-gate' ? sigmoid(value) : value * sigmoid(value) * (split ? 1.25 : 1);
          if (limit !== null) result = Math.min(limit, Math.max(-limit, result));
          return round(result);
        });
        const buffers = [];
        const upload = data => {
          const encoded = dtype === 'f32' ? new Float32Array(data) : Uint16Array.from(data, float32ToFloat16);
          const buffer = acquireBuffer(encoded.byteLength); buffers.push(buffer); uploadData(buffer, encoded);
          return buffer;
        };
        try {
          const input = createTensor(upload(inputValues), dtype, split ? [numTokens, 2 * dim] : [values.length]);
          const gate = kind === 'sigmoid-gate' ? createTensor(upload(values), dtype, [values.length]) : null;
          for (const recorded of [false, true]) {
            const recorder = recorded ? createCommandRecorder('unclamped-activation') : null;
            let result;
            try {
              if (split) {
                const options = { numTokens, dim, activation: 'silu', swigluLimit: limit };
                result = recorder ? await recordSiLURowSplit(recorder, input, options) : await runSiLURowSplit(input, options);
              } else {
                const options = { size: values.length, swigluLimit: null, useVec4: kind === 'unary-vec4',
                  gate, gateActivation: 'sigmoid', inputActivation: gate ? 'identity' : 'silu' };
                result = recorder ? await recordSiLU(recorder, input, options) : await runSiLU(input, options);
              }
              if (recorder) await recorder.submitAndWait();
              const raw = await readBuffer(result.buffer, references.length * (dtype === 'f32' ? 4 : 2));
              const actual = dtype === 'f32' ? Array.from(new Float32Array(raw)) : Array.from(new Uint16Array(raw), float16ToFloat32);
              const failures = [];
              for (let i = 0; i < references.length; i++) {
                const expected = references[i], value = actual[i];
                // F16 permits subnormal flushing; otherwise allow two rounding
                // units around an independent rounded mathematical reference.
                const flushed = dtype === 'f16' && Math.abs(expected) < 2 ** -14 && value === 0;
                const allowance = dtype === 'f32' ? Math.abs(expected) * 8 * 2 ** -23 + 1e-40
                  : Math.abs(expected) * 2 * 2 ** -10 + 2 ** -24;
                let valid = Number.isFinite(value) && (flushed || Math.abs(value - expected) <= allowance);
                if (!split && values[i] >= 32) valid &&= value === expected;
                if (!valid) failures.push({ index: i, input: values[i % values.length], actual: value, expected });
              }
              receipt.rows.push({ dtype, kind, recorded, count: references.length,
                violations: failures.length, firstFailures: failures.slice(0, 4),
                outputSha256: hash(Buffer.from(raw)) });
            } finally {
              if (result) releaseBuffer(result.buffer);
              recorder?.abort();
            }
          }
        } finally {
          await device.queue.onSubmittedWorkDone();
          for (const buffer of buffers) releaseBuffer(buffer);
        }
      }
    }
  } finally {
    destroyDevice();
    receipt.passed = receipt.rows.length === 20 && errors.length === 0 && receipt.rows.every(row => row.violations === 0);
    if (outputPath) await writeFile(outputPath, JSON.stringify(receipt, null, 2) + '\n', { flag: 'wx' });
  }
  console.log(JSON.stringify({ passed: receipt.passed, cases: receipt.rows.length,
    violations: receipt.rows.reduce((sum, row) => sum + row.violations, 0), errors }));
  assert(receipt.passed, 'Activation input must remain unclamped; only explicit swigluLimit clips output.');
}
