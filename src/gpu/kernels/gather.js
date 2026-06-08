import { getKernelCapabilities } from '../device.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { WORKGROUP_SIZES, VEC4_ELEMENTS_PER_WG } from './constants.js';
import { unifiedKernelWrapper } from './utils.js';
import { trace } from '../../debug/index.js';
import { createTensor } from '../tensor.js';
import { DTYPE_SIZES, padToQ4KBlock } from '../../config/schema/index.js';
import { selectRuleValue as selectKernelRuleValue } from './rule-registry.js';
import { selectRuleValue as selectSharedRuleValue } from '../../rules/rule-registry.js';

const SPLIT_GATHER_SECTION_COUNT = 4;

function selectGatherVariant(useF16Input, useF16Output, useVec4, useLiteRTInt4Input = false) {
  return selectKernelRuleValue(
    'gather',
    'variant',
    { useF16Input, useF16Output, useVec4, useLiteRTInt4Input }
  );
}

function resolveLiteRTInt4StorageEncoding(storageEncoding) {
  const normalized = String(storageEncoding ?? '').toLowerCase();
  if (normalized !== 'signed' && normalized !== 'offset_binary') {
    throw new Error(
      `[Gather] LiteRT INT4 embeddings require storageEncoding "signed" or "offset_binary", ` +
      `got "${normalized || 'missing'}".`
    );
  }
  return normalized;
}

function normalizeSplitGatherSections(splitEmbedding) {
  const sections = splitEmbedding?.sections;
  if (!Array.isArray(sections) || sections.length === 0) {
    throw new Error('[Gather] split embeddings require at least one GPU section.');
  }
  if (sections.length > SPLIT_GATHER_SECTION_COUNT) {
    throw new Error(
      `[Gather] split embeddings have ${sections.length} sections; ` +
      `gather_split4 supports at most ${SPLIT_GATHER_SECTION_COUNT}.`
    );
  }
  let nextRowStart = 0;
  for (let i = 0; i < sections.length; i++) {
    const section = sections[i];
    if (!section?.buffer) {
      throw new Error(`[Gather] split section ${i} is missing a GPU buffer.`);
    }
    if (section.rowStart !== nextRowStart) {
      throw new Error(
        `[Gather] split section ${i} rowStart=${section.rowStart} is not contiguous from row ${nextRowStart}.`
      );
    }
    if (!Number.isInteger(section.rowCount) || section.rowCount <= 0) {
      throw new Error(`[Gather] split section ${i} has invalid rowCount.`);
    }
    nextRowStart += section.rowCount;
  }
  const padded = [...sections];
  while (padded.length < SPLIT_GATHER_SECTION_COUNT) {
    padded.push({ buffer: sections[0].buffer, rowStart: nextRowStart, rowCount: 0 });
  }
  return padded;
}

async function _gatherSplit4(
  target,
  indices,
  splitEmbedding,
  numTokens,
  hiddenSize,
  vocabSize,
  options = {}
) {
  const {
    outputBuffer = null,
    embeddingDtype,
    outputDtype,
    transpose = false,
    indexOffset = 0,
    inputHiddenSize = hiddenSize,
    hiddenOffset = 0,
    indirectBuffer = null,
    indirectOffset = 0,
  } = options;

  if (embeddingDtype == null) {
    throw new Error('[Gather] embeddingDtype is required.');
  }
  if (outputDtype == null) {
    throw new Error('[Gather] outputDtype is required.');
  }
  const sections = normalizeSplitGatherSections(splitEmbedding);
  const logicalRows = sections.reduce((sum, section) => sum + section.rowCount, 0);
  if (logicalRows < vocabSize) {
    throw new Error(
      `[Gather] split embeddings expose ${logicalRows} rows but vocabSize=${vocabSize}.`
    );
  }

  const caps = getKernelCapabilities();
  if (!caps.hasF16) {
    throw new Error('[Gather] gather_split4 requires shader-f16 support.');
  }
  if (splitEmbedding?.dtype !== 'f16' || embeddingDtype !== 'f16') {
    throw new Error('[Gather] gather_split4 requires f16 split embeddings.');
  }
  if (outputDtype !== 'f16') {
    throw new Error('[Gather] gather_split4 requires outputDtype=f16.');
  }
  if (splitEmbedding?.layout !== 'row' || transpose) {
    throw new Error('[Gather] gather_split4 requires row-major embeddings.');
  }
  if (hiddenSize % 4 !== 0) {
    throw new Error('[Gather] gather_split4 requires hiddenSize to be divisible by 4.');
  }
  if (!Number.isFinite(inputHiddenSize) || inputHiddenSize < hiddenSize) {
    throw new Error('[Gather] inputHiddenSize must be >= hiddenSize.');
  }
  if (!Number.isFinite(hiddenOffset) || hiddenOffset < 0 || (hiddenOffset + hiddenSize) > inputHiddenSize) {
    throw new Error('[Gather] hiddenOffset must select a valid hidden slice inside inputHiddenSize.');
  }

  trace.embed(
    `GatherSplit4: numTokens=${numTokens}, hiddenSize=${hiddenSize}, vocabSize=${vocabSize}, ` +
    `indexOffset=${indexOffset}, inputHiddenSize=${inputHiddenSize}, hiddenOffset=${hiddenOffset}, ` +
    `sections=${splitEmbedding.sections.length}, embeddingDtype=${embeddingDtype}, outputDtype=${outputDtype}`
  );

  const actualDtype = 'f16';
  const bytesPerElement = DTYPE_SIZES[actualDtype];
  const paddedHiddenSize = padToQ4KBlock(hiddenSize);
  const outputSize = numTokens * paddedHiddenSize * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'gather_split4_output');
  const ownedOutput = outputBuffer ? null : output;

  const uniforms = {
    num_tokens: numTokens,
    hidden_size: hiddenSize,
    vocab_size: vocabSize,
    index_offset: indexOffset,
    input_hidden_size: inputHiddenSize,
    hidden_offset: hiddenOffset,
    section0_rows: sections[0].rowCount,
    section1_rows: sections[1].rowCount,
    section2_rows: sections[2].rowCount,
    section3_rows: sections[3].rowCount,
    _pad0: 0,
    _pad1: 0,
  };

  const workgroups = indirectBuffer
    ? { indirectBuffer, indirectOffset }
    : Math.ceil((numTokens * hiddenSize) / VEC4_ELEMENTS_PER_WG);

  try {
    await unifiedKernelWrapper(
      'gather_split4',
      target,
      'f16_vec4_f16_out',
      [
        indices,
        sections[0].buffer,
        sections[1].buffer,
        sections[2].buffer,
        sections[3].buffer,
        output,
      ],
      uniforms,
      workgroups
    );
    return createTensor(output, actualDtype, [numTokens, hiddenSize], 'gather_split4_output');
  } catch (error) {
    if (ownedOutput) {
      releaseBuffer(ownedOutput);
    }
    throw error;
  }
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
    outputBuffer = null,
    embeddingDtype,
    outputDtype,
    transpose = false,
    indexOffset = 0,
    inputHiddenSize = hiddenSize,
    hiddenOffset = 0,
    indirectBuffer = null,
    indirectOffset = 0,
    storageEncoding = null,
  } = options;

  const caps = getKernelCapabilities();
  if (embeddingDtype == null) {
    throw new Error('[Gather] embeddingDtype is required.');
  }
  if (outputDtype == null) {
    throw new Error('[Gather] outputDtype is required.');
  }
  const useLiteRTInt4Input = embeddingDtype === 'litert_int4';
  if (embeddingDtype !== 'f16' && embeddingDtype !== 'f32' && !useLiteRTInt4Input) {
    throw new Error(`[Gather] unsupported embeddingDtype="${embeddingDtype}".`);
  }
  if (embeddingDtype === 'f16' && !caps.hasF16) {
    throw new Error('[Gather] embeddingDtype=f16 requires shader-f16 support.');
  }
  if (outputDtype === 'f16' && !caps.hasF16) {
    throw new Error('[Gather] outputDtype=f16 requires shader-f16 support.');
  }

  const usesHiddenSlice = inputHiddenSize !== hiddenSize || hiddenOffset !== 0;
  const requestedVec4 = usesHiddenSlice ? false : options.useVec4;
  const wantsVec4 = requestedVec4 ?? true;
  if (requestedVec4 === true && hiddenSize % 4 !== 0) {
    throw new Error('[Gather] useVec4=true requires hiddenSize to be divisible by 4.');
  }
  if (!Number.isFinite(inputHiddenSize) || inputHiddenSize < hiddenSize) {
    throw new Error('[Gather] inputHiddenSize must be >= hiddenSize.');
  }
  if (!Number.isFinite(hiddenOffset) || hiddenOffset < 0 || (hiddenOffset + hiddenSize) > inputHiddenSize) {
    throw new Error('[Gather] hiddenOffset must select a valid hidden slice inside inputHiddenSize.');
  }

  const useF16Input = embeddingDtype === 'f16';
  const useF16Output = outputDtype === 'f16';
  const useVec4 = wantsVec4 && hiddenSize % 4 === 0;
  if (useLiteRTInt4Input) {
    resolveLiteRTInt4StorageEncoding(storageEncoding);
    if (outputDtype !== 'f16' && outputDtype !== 'f32') {
      throw new Error('[Gather] LiteRT INT4 embeddings require outputDtype=f16 or outputDtype=f32.');
    }
    if (transpose) {
      throw new Error('[Gather] LiteRT INT4 embeddings require row-major layout.');
    }
    if (usesHiddenSlice) {
      throw new Error('[Gather] LiteRT INT4 embeddings do not support hidden slicing.');
    }
    if (!useVec4) {
      throw new Error('[Gather] LiteRT INT4 embeddings require vec4 gather; hiddenSize must be divisible by 4.');
    }
  }

  trace.embed(
    `Gather: numTokens=${numTokens}, hiddenSize=${hiddenSize}, vocabSize=${vocabSize}, ` +
    `transpose=${transpose}, indexOffset=${indexOffset}, ` +
    `inputHiddenSize=${inputHiddenSize}, hiddenOffset=${hiddenOffset}, ` +
    `embeddingDtype=${embeddingDtype}, outputDtype=${outputDtype}, ` +
    `useF16Input=${useF16Input}, useF16Output=${useF16Output}`
  );

  const variant = selectGatherVariant(useF16Input, useF16Output, useVec4, useLiteRTInt4Input);
  trace.embed(`Gather variant: ${variant}`);
  const constants = useLiteRTInt4Input
    ? { STORAGE_OFFSET_BINARY: resolveLiteRTInt4StorageEncoding(storageEncoding) === 'offset_binary' ? 1 : 0 }
    : null;

  // Pad hiddenSize to Q4K alignment for downstream fused Q4K matmul kernels
  // that read 256-element blocks. Extra padding elements stay zero.
  const actualDtype = selectSharedRuleValue('shared', 'dtype', 'f16OrF32', { useF16: useF16Output });
  const bytesPerElement = DTYPE_SIZES[actualDtype];
  const paddedHiddenSize = padToQ4KBlock(hiddenSize);
  const outputSize = numTokens * paddedHiddenSize * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'gather_output');
  const ownedOutput = outputBuffer ? null : output;

  const uniforms = {
    num_tokens: numTokens,
    hidden_size: hiddenSize,
    vocab_size: vocabSize,
    transpose: transpose ? 1 : 0,
    index_offset: indexOffset,
    input_hidden_size: inputHiddenSize,
    hidden_offset: hiddenOffset,
    _pad0: 0,
    _pad1: 0,
  };

  const workgroups = indirectBuffer
    ? { indirectBuffer, indirectOffset }
    : (useVec4
      ? Math.ceil((numTokens * hiddenSize) / VEC4_ELEMENTS_PER_WG)
      : Math.ceil((numTokens * hiddenSize) / WORKGROUP_SIZES.DEFAULT));

  try {
    await unifiedKernelWrapper(
      'gather',
      target,
      variant,
      [indices, embeddings, output],
      uniforms,
      workgroups,
      constants
    );
    return createTensor(output, actualDtype, [numTokens, hiddenSize], 'gather_output');
  } catch (error) {
    if (ownedOutput) {
      releaseBuffer(ownedOutput);
    }
    throw error;
  }
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

export async function runGatherSplit4(
  indices,
  splitEmbedding,
  numTokens,
  hiddenSize,
  vocabSize,
  options = {}
) {
  return _gatherSplit4(null, indices, splitEmbedding, numTokens, hiddenSize, vocabSize, options);
}

export async function recordGatherSplit4(
  recorder,
  indices,
  splitEmbedding,
  numTokens,
  hiddenSize,
  vocabSize,
  options = {}
) {
  return _gatherSplit4(recorder, indices, splitEmbedding, numTokens, hiddenSize, vocabSize, options);
}
