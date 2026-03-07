import { getDevice } from '../device.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { unifiedKernelWrapper } from './utils.js';
import { selectRuleValue } from './rule-registry.js';
import { WORKGROUP_SIZES } from './constants.js';

function selectSanaLinearAttentionVariant(isF16) {
  return selectRuleValue('sanaLinearAttention', 'variant', { isF16 });
}

async function runSummary(target, query, key, value, summaryBuffer, uniforms, variant) {
  const summarySize = uniforms.num_heads * (uniforms.head_dim + 1) * uniforms.head_dim;
  await unifiedKernelWrapper(
    'sana_linear_attention_summary',
    target,
    variant,
    [query, key, value, summaryBuffer],
    {
      num_heads: uniforms.num_heads,
      head_dim: uniforms.head_dim,
      num_tokens: uniforms.num_tokens,
      hidden_size: uniforms.hidden_size,
      _pad0: 0,
      _pad1: 0,
    },
    Math.ceil(summarySize / WORKGROUP_SIZES.DEFAULT)
  );
}

async function runApply(target, query, summaryBuffer, outputBuffer, uniforms, variant) {
  await unifiedKernelWrapper(
    'sana_linear_attention_apply',
    target,
    variant,
    [query, summaryBuffer, outputBuffer],
    {
      num_heads: uniforms.num_heads,
      head_dim: uniforms.head_dim,
      num_tokens: uniforms.num_tokens,
      hidden_size: uniforms.hidden_size,
      eps: uniforms.eps,
      _pad0: 0,
      _pad1: 0,
      _pad2: 0,
    },
    [Math.ceil(uniforms.hidden_size / WORKGROUP_SIZES.DEFAULT), uniforms.num_tokens, 1]
  );
}

async function _sanaLinearAttention(target, query, key, value, options = {}) {
  const recorder = target && typeof target.beginComputePass === 'function' ? target : null;
  const device = target?.device || getDevice();
  if (!device) {
    throw new Error('SanaLinearAttention requires a WebGPU device.');
  }

  const {
    numHeads,
    headDim,
    numTokens = query.shape?.[0],
    hiddenSize = query.shape?.[1],
    eps = 1e-15,
    outputBuffer = null,
    summaryBuffer = null,
  } = options;
  const ownsSummary = summaryBuffer == null;
  const ownsOutput = outputBuffer == null;

  if (
    !Number.isFinite(numHeads) ||
    !Number.isFinite(headDim) ||
    !Number.isFinite(numTokens) ||
    !Number.isFinite(hiddenSize)
  ) {
    throw new Error('SanaLinearAttention requires numHeads, headDim, numTokens, and hiddenSize.');
  }
  if (hiddenSize !== numHeads * headDim) {
    throw new Error(`SanaLinearAttention hiddenSize mismatch: ${hiddenSize} != ${numHeads} * ${headDim}`);
  }

  const isF16 = query.dtype === 'f16';
  const variant = selectSanaLinearAttentionVariant(isF16);
  const temporarySummary = summaryBuffer || acquireBuffer(
    numHeads * (headDim + 1) * headDim * Float32Array.BYTES_PER_ELEMENT,
    undefined,
    'sana_linear_attention_summary'
  );
  const output = outputBuffer || acquireBuffer(
    numTokens * hiddenSize * dtypeBytes(query.dtype),
    undefined,
    'sana_linear_attention_output'
  );

  const uniforms = {
    num_heads: numHeads,
    head_dim: headDim,
    num_tokens: numTokens,
    hidden_size: hiddenSize,
    eps,
  };

  try {
    await runSummary(target, query, key, value, temporarySummary, uniforms, variant);
    await runApply(target, query, temporarySummary, output, uniforms, variant);
    return createTensor(output, query.dtype, [numTokens, hiddenSize], 'sana_linear_attention_output');
  } catch (error) {
    if (ownsOutput) {
      releaseBuffer(output);
    }
    throw error;
  } finally {
    if (ownsSummary) {
      if (recorder) {
        recorder.trackTemporaryBuffer(temporarySummary);
      } else {
        releaseBuffer(temporarySummary);
      }
    }
  }
}

export async function runSanaLinearAttention(query, key, value, options = {}) {
  return _sanaLinearAttention(null, query, key, value, options);
}

export async function recordSanaLinearAttention(recorder, query, key, value, options = {}) {
  return _sanaLinearAttention(recorder, query, key, value, options);
}
