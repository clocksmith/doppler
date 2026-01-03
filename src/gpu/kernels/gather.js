import { getKernelCapabilities } from '../device.js';
import { acquireBuffer } from '../../memory/buffer-pool.js';
import { WORKGROUP_SIZES, VEC4_ELEMENTS_PER_WG } from './constants.js';
import { unifiedKernelWrapper } from './utils.js';
import { trace } from '../../debug/index.js';
import { createTensor } from '../tensor.js';
import { DTYPE_SIZES, padToQ4KBlock } from '../../config/schema/index.js';
import { selectRuleValue as selectKernelRuleValue } from './rule-registry.js';
import { selectRuleValue as selectSharedRuleValue } from '../../rules/rule-registry.js';

function selectGatherVariant(useF16Input, useF16Output, useVec4) {
  return selectKernelRuleValue(
    'gather',
    'variant',
    { useF16Input, useF16Output, useVec4 }
  );
}

async function _gather(
  target,
  indices,
  embeddings,
  numTokens,
  hiddenSize,
  vocabSize,
  options = {}
) {
  const {
    useVec4 = true,
    outputBuffer = null,
    embeddingDtype,
    outputDtype,
    transpose = false,
    indexOffset = 0,
    indirectBuffer = null,
    indirectOffset = 0,
  } = options;

  const caps = getKernelCapabilities();
  if (embeddingDtype == null) {
    throw new Error('[Gather] embeddingDtype is required.');
  }
  if (outputDtype == null) {
    throw new Error('[Gather] outputDtype is required.');
  }

  const useF16Input = embeddingDtype === 'f16' && caps.hasF16;
  const useF16Output = outputDtype === 'f16' && caps.hasF16;

  trace.embed(
    `Gather: numTokens=${numTokens}, hiddenSize=${hiddenSize}, vocabSize=${vocabSize}, ` +
    `transpose=${transpose}, indexOffset=${indexOffset}, ` +
    `embeddingDtype=${embeddingDtype}, outputDtype=${outputDtype}, ` +
    `useF16Input=${useF16Input}, useF16Output=${useF16Output}`
  );

  const variant = selectGatherVariant(useF16Input, useF16Output, useVec4);
  trace.embed(`Gather variant: ${variant}`);

  // Pad hiddenSize to Q4K alignment for downstream fused Q4K matmul kernels
  // that read 256-element blocks. Extra padding elements stay zero.
  const actualDtype = selectSharedRuleValue('shared', 'dtype', 'f16OrF32', { useF16: useF16Output });
  const bytesPerElement = DTYPE_SIZES[actualDtype];
  const paddedHiddenSize = padToQ4KBlock(hiddenSize);
  const outputSize = numTokens * paddedHiddenSize * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'gather_output');

  const uniforms = {
    num_tokens: numTokens,
    hidden_size: hiddenSize,
    vocab_size: vocabSize,
    transpose: transpose ? 1 : 0,
    index_offset: indexOffset,
    _pad0: 0,
    _pad1: 0,
    _pad2: 0,
  };

  const workgroups = indirectBuffer
    ? { indirectBuffer, indirectOffset }
    : (useVec4
      ? Math.ceil((numTokens * hiddenSize) / VEC4_ELEMENTS_PER_WG)
      : Math.ceil((numTokens * hiddenSize) / WORKGROUP_SIZES.DEFAULT));

  await unifiedKernelWrapper(
    'gather',
    target,
    variant,
    [indices, embeddings, output],
    uniforms,
    workgroups
  );

  return createTensor(output, actualDtype, [numTokens, hiddenSize], 'gather_output');
}

export async function runGather(
  indices,
  embeddings,
  numTokens,
  hiddenSize,
  vocabSize,
  options = {}
) {
  return _gather(null, indices, embeddings, numTokens, hiddenSize, vocabSize, options);
}

export async function recordGather(
  recorder,
  indices,
  embeddings,
  numTokens,
  hiddenSize,
  vocabSize,
  options = {}
) {
  return _gather(recorder, indices, embeddings, numTokens, hiddenSize, vocabSize, options);
}

