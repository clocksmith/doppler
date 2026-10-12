import { getDevice } from '../device.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { unifiedKernelWrapper, withKernelOutput } from './kernel-execution.js';
import { planRoPEPrecomputeDispatch } from './rope-dispatch.js';
import { validateRoPEInverseFrequencies } from '../../config/rope-frequencies.js';

/** @type {typeof import('./paired-rope.js').runPairedRoPEPrecompute} */
export async function runPairedRoPEPrecompute(options) {
  const { theta, ropeScale, rotaryDim, frequencyBaseDim, maxSeqLen } = options;
  const positionCount = options.positionCount ?? maxSeqLen;
  if (![theta, ropeScale].every(value => Number.isFinite(value) && value > 0 && Number.isFinite(Math.fround(value)) && Math.fround(value) !== 0)) {
    throw new Error('[PairedRoPE] Positive finite theta and scale required.');
  }
  if (![rotaryDim, frequencyBaseDim, maxSeqLen, positionCount].every(value => Number.isSafeInteger(value) && value > 0 && value <= 0xffffff)
    || rotaryDim % 2 !== 0 || frequencyBaseDim % 2 !== 0 || rotaryDim > frequencyBaseDim || positionCount > maxSeqLen) {
    throw new Error('[PairedRoPE] Explicit even rotary geometry and bounded exact position extent required.');
  }
  if ((options.scalingType != null && options.scalingType !== 'linear') || options.positionPlan != null || options.mropeSection != null) {
    throw new Error('[PairedRoPE] This declared paired path supports ordinary or linear scaling only.');
  }
  const frequencies = validateRoPEInverseFrequencies(options.inverseFrequencies ?? null, rotaryDim, 'Paired RoPE inverseFrequencies');
  // Source configuration scalars are encoded as hi/lo bytes; tensor computation stays in WGSL.
  const encoded = new Float32Array(Math.max(2, (frequencies?.length ?? 0) * 2));
  frequencies?.forEach((value, index) => { encoded[index * 2] = Math.fround(value); encoded[index * 2 + 1] = Math.fround(value - encoded[index * 2]); });
  const count = positionCount * rotaryDim / 2;
  if (!Number.isSafeInteger(count) || count > 0xffffffff) throw new Error('[PairedRoPE] Frequency extent exceeds U32 indexing.');
  const device = getDevice();
  if (!device) throw new Error('[PairedRoPE] GPU device required.');
  const dispatch = planRoPEPrecomputeDispatch(device, count);
  const data = acquireBuffer(encoded.byteLength, undefined, 'paired_rope_source');
  try {
    device.queue.writeBuffer(data, 0, encoded);
    return await withKernelOutput(null, null, count * 8, 'paired_rope_cos', async cos =>
      withKernelOutput(null, null, count * 8, 'paired_rope_sin', async sin => {
        await unifiedKernelWrapper('rope_precompute', null, 'paired', [data, cos, sin], {
          position_count: positionCount, rotary_dim: rotaryDim, frequency_base_dim: frequencyBaseDim,
          dispatch_stride: dispatch.dispatchStride,
          theta_hi: Math.fround(theta), theta_lo: Math.fround(theta - Math.fround(theta)),
          scale_hi: Math.fround(ropeScale), scale_lo: Math.fround(ropeScale - Math.fround(ropeScale)), inverse_supplied: frequencies !== null ? 1 : 0,
        }, dispatch.workgroups);
        return { cos, sin };
      }));
  } finally { releaseBuffer(data); }
}
