
import { getKernelCapabilities } from '../device.js';
import { createTensor } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { unifiedKernelWrapper } from './utils.js';
import { getKernelThresholds } from '../../config/schema/index.js';
import { selectRuleValue } from './rule-registry.js';

const getRopeDefaults = () => getKernelThresholds().rope;

async function _rope(target, input, freqsCos, freqsSin, seqLen, options = {}) {
  const ropeDefaults = getRopeDefaults();
  const {
    numHeads = 1,
    headDim = 64,
    ropeTheta = ropeDefaults.defaultTheta,
  } = options;

  if (headDim % 2 !== 0) {
    throw new Error(`RoPE headDim must be even, got ${headDim}`);
  }

  const caps = getKernelCapabilities();
  const useF16 = input.dtype === 'f16' && caps.hasF16;
  const variant = selectRuleValue('rope', 'variant', { useF16 });

  const halfDim = headDim / 2;
  const workgroups = Math.ceil((seqLen * numHeads * halfDim) / WORKGROUP_SIZES.DEFAULT);

  await unifiedKernelWrapper(
    'rope', target, variant,
    [input, freqsCos, freqsSin],
    {
      seq_len: seqLen,
      num_heads: numHeads,
      head_dim: headDim,
      start_pos: options.startPos ?? ropeDefaults.defaultStartPos,
      rope_base: ropeTheta,
      rope_scale: 1.0,
    },
    workgroups
  );

  return createTensor(input.buffer, input.dtype, [...input.shape], 'rope_output');
}

export async function runRoPE(input, freqsCos, freqsSin, seqLen, options = {}) {
  return _rope(null, input, freqsCos, freqsSin, seqLen, options);
}

export async function recordRoPE(recorder, input, freqsCos, freqsSin, seqLen, options = {}) {
  return _rope(recorder, input, freqsCos, freqsSin, seqLen, options);
}
