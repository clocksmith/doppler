
import { getKernelCapabilities } from '../device.js';
import { createTensor } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { unifiedKernelWrapper } from './utils.js';
import { getKernelThresholds } from '../../config/schema/index.js';
import { selectRuleValue } from './rule-registry.js';
import { castF16ToF32, castF32ToF16, recordCastF16ToF32, recordCastF32ToF16 } from './cast.js';
import { releaseBuffer } from '../../memory/buffer-pool.js';

const getRopeDefaults = () => getKernelThresholds().rope;

async function _rope(target, input, freqsCos, freqsSin, seqLen, options = {}) {
  const ropeDefaults = getRopeDefaults();
  const {
    numHeads = 1,
    headDim = 64,
    rotaryDim = headDim,
    interleaved = false,
    ropeTheta = ropeDefaults.defaultTheta,
  } = options;

  if (headDim % 2 !== 0) {
    throw new Error(`RoPE headDim must be even, got ${headDim}`);
  }
  if (rotaryDim % 2 !== 0) {
    throw new Error(`RoPE rotaryDim must be even, got ${rotaryDim}`);
  }
  if (rotaryDim <= 0 || rotaryDim > headDim) {
    throw new Error(`RoPE rotaryDim must be in (0, headDim]; got ${rotaryDim} for headDim ${headDim}`);
  }

  // F16 RoPE kernel does not support interleaved or partial rotation.
  // When the execution graph pairs f16 projections with f32 RoPE, bridge
  // with a cast round-trip: f16→f32, run f32 variant, f32→f16 back into
  // the original buffer to preserve the in-place contract.
  const needsF32Cast = input.dtype === 'f16' && (rotaryDim !== headDim || interleaved);
  let ropeInput = input;
  let f32TempBuffer = null;

  if (needsF32Cast) {
    ropeInput = target
      ? await recordCastF16ToF32(target, input)
      : await castF16ToF32(input);
    f32TempBuffer = ropeInput.buffer;
  }

  const caps = getKernelCapabilities();
  const useF16 = ropeInput.dtype === 'f16' && caps.hasF16;
  const variant = selectRuleValue('rope', 'variant', { useF16 });

  const halfDim = rotaryDim / 2;
  const workgroups = Math.ceil((seqLen * numHeads * halfDim) / WORKGROUP_SIZES.DEFAULT);

  await unifiedKernelWrapper(
    'rope', target, variant,
    [ropeInput, freqsCos, freqsSin],
    {
      seq_len: seqLen,
      num_heads: numHeads,
      head_dim: headDim,
      rotary_dim: rotaryDim,
      start_pos: options.startPos ?? ropeDefaults.defaultStartPos,
      rope_base: ropeTheta,
      rope_scale: 1.0,
      interleaved: interleaved ? 1 : 0,
    },
    workgroups
  );

  if (needsF32Cast) {
    target
      ? await recordCastF32ToF16(target, ropeInput, { outputBuffer: input.buffer })
      : await castF32ToF16(ropeInput, { outputBuffer: input.buffer });
    if (target && typeof target.trackTemporaryBuffer === 'function') {
      target.trackTemporaryBuffer(f32TempBuffer);
    } else {
      releaseBuffer(f32TempBuffer);
    }
  }

  return createTensor(input.buffer, input.dtype, [...input.shape], 'rope_output');
}

export async function runRoPE(input, freqsCos, freqsSin, seqLen, options = {}) {
  return _rope(null, input, freqsCos, freqsSin, seqLen, options);
}

export async function recordRoPE(recorder, input, freqsCos, freqsSin, seqLen, options = {}) {
  return _rope(recorder, input, freqsCos, freqsSin, seqLen, options);
}
