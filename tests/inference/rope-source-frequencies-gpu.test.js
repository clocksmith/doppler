import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { destroyDevice } from '../../src/gpu/device.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import { runRoPEPrecompute } from '../../src/gpu/kernels/rope-precompute.js';
import { readBuffer, releaseBuffer } from '../../src/memory/buffer-pool.js';

const FREQUENCY_TOLERANCE = 0.00001;
const SEQUENCE_LENGTH = 111;
const source = [1, 0.464111328125, 0.2154541015625, 0.0999755859375,
  0.046417236328125, 0.02154541015625, 0.01000213623046875, 0.004642486572265625,
  0.002155303955078125, 0.0010004043579101562, 0.0004642009735107422, 0.00021541118621826172];
const probe = await probeNodeGPU({ installFileFetchShim: true });
if (!probe.ready) {
  console.log(`rope-source-frequencies-gpu.test: skipped (${probe.reason})`);
} else {
  try {
    for (const inverseFrequencies of [null, source]) {
      const options = { theta: 10000, rotaryDim: 24, frequencyBaseDim: 24,
        maxSeqLen: SEQUENCE_LENGTH, ropeScale: 1, scalingType: null, scaling: null, inverseFrequencies };
      const tables = await runRoPEPrecompute(options);
      try {
        for (const name of ['cos', 'sin']) {
          const values = new Float32Array(await readBuffer(tables[name], SEQUENCE_LENGTH * source.length * 4));
          for (let index = 0; index < values.length; index++) {
            const dimension = index % source.length;
            const position = Math.floor(index / source.length);
            const frequency = inverseFrequencies?.[dimension] ?? 1 / 10000 ** (2 * dimension / 24);
            const expected = Math[name](Math.fround(position * frequency));
            assert(Math.abs(values[index] - expected) < FREQUENCY_TOLERANCE,
              `${name}/${inverseFrequencies === null ? 'generated' : 'source'}/${position}/${dimension}: ${values[index]} != ${expected}`);
          }
        }
      } finally { releaseBuffer(tables.cos); releaseBuffer(tables.sin); }
    }
    console.log('rope-source-frequencies-gpu.test: ok (generated and retained source frequencies)');
  } finally { destroyDevice(); releaseNodeWebGPU(); }
}
