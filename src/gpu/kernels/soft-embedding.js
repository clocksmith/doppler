import { getDevice } from '../device.js';
import { createTensor } from '../tensor.js';
import { isSplitWeightBuffer } from '../weight-buffer.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createPipeline, createUniformBufferWithView, createBindGroupWithValidation } from './utils.js';
import { dispatchKernel } from './dispatch.js';

const UNIFORM_SIZE = 32;
const WORKGROUP_X = 8;
const WORKGROUP_Y = 8;

function validatePositiveInt(value, label) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`[SoftEmbedding] ${label} must be a positive integer; got ${String(value)}.`);
  }
}

function createSectionUniformBuffer(device, section, sectionIndex, numTokens, hiddenSize, vocabSize) {
  return createUniformBufferWithView(
    'soft_embedding_split_uniforms',
    UNIFORM_SIZE,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, hiddenSize, true);
      view.setUint32(8, vocabSize, true);
      view.setUint32(12, section.rowStart, true);
      view.setUint32(16, section.rowCount, true);
      view.setUint32(20, sectionIndex === 0 ? 0 : 1, true);
      view.setUint32(24, 0, true);
      view.setUint32(28, 0, true);
    },
    null,
    device
  );
}

function validateSplitSections(splitEmbedding, vocabSize) {
  const sections = splitEmbedding.sections;
  if (!Array.isArray(sections) || sections.length === 0) {
    throw new Error('[SoftEmbedding] split embedding requires at least one section.');
  }
  let nextRowStart = 0;
  for (let sectionIndex = 0; sectionIndex < sections.length; sectionIndex += 1) {
    const section = sections[sectionIndex];
    if (!section?.buffer) {
      throw new Error(`[SoftEmbedding] split section ${sectionIndex} is missing a GPU buffer.`);
    }
    if (section.rowStart !== nextRowStart) {
      throw new Error(
        `[SoftEmbedding] split section ${sectionIndex} rowStart=${section.rowStart} ` +
        `is not contiguous from row ${nextRowStart}.`
      );
    }
    if (!Number.isInteger(section.rowCount) || section.rowCount <= 0) {
      throw new Error(`[SoftEmbedding] split section ${sectionIndex} has invalid rowCount.`);
    }
    nextRowStart += section.rowCount;
  }
  if (nextRowStart !== vocabSize) {
    throw new Error(
      `[SoftEmbedding] split embedding exposes ${nextRowStart} rows but vocabSize=${vocabSize}.`
    );
  }
}

async function dispatchSplitSection({
  device,
  pipeline,
  softmaxTensor,
  section,
  sectionIndex,
  output,
  numTokens,
  hiddenSize,
  vocabSize,
}) {
  const uniformBuffer = createSectionUniformBuffer(
    device,
    section,
    sectionIndex,
    numTokens,
    hiddenSize,
    vocabSize
  );

  try {
    const bindGroup = await createBindGroupWithValidation(device, {
      label: 'soft_embedding_split_bind_group',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: softmaxTensor.buffer } },
        { binding: 2, resource: { buffer: section.buffer } },
        { binding: 3, resource: { buffer: output } },
      ],
    }, 'soft_embedding/split_f16');

    dispatchKernel(
      null,
      pipeline,
      bindGroup,
      [Math.ceil(hiddenSize / WORKGROUP_X), Math.ceil(numTokens / WORKGROUP_Y), 1],
      'soft_embedding_split_f16'
    );
  } finally {
    uniformBuffer.destroy();
  }
}

export async function runSoftEmbeddingSplitF16(softmaxTensor, splitEmbedding, numTokens, hiddenSize, vocabSize, options = {}) {
  validatePositiveInt(numTokens, 'numTokens');
  validatePositiveInt(hiddenSize, 'hiddenSize');
  validatePositiveInt(vocabSize, 'vocabSize');

  if (softmaxTensor?.dtype !== 'f32') {
    throw new Error(`[SoftEmbedding] split f16 path requires f32 softmax input, got "${softmaxTensor?.dtype ?? 'missing'}".`);
  }
  if (!isSplitWeightBuffer(splitEmbedding)) {
    throw new Error('[SoftEmbedding] split f16 path requires a SplitWeightBuffer embedding table.');
  }
  if (splitEmbedding.dtype !== 'f16' || splitEmbedding.layout !== 'row') {
    throw new Error(
      `[SoftEmbedding] split path supports row-major f16 embeddings only; ` +
      `got dtype=${splitEmbedding.dtype}, layout=${splitEmbedding.layout}.`
    );
  }
  validateSplitSections(splitEmbedding, vocabSize);
  const shape = Array.isArray(splitEmbedding.shape) ? splitEmbedding.shape : null;
  if (!shape || shape.length !== 2 || shape[0] !== vocabSize || shape[1] !== hiddenSize) {
    throw new Error(
      `[SoftEmbedding] split embedding shape mismatch: expected [${vocabSize}, ${hiddenSize}], ` +
      `got ${shape ? `[${shape.join(', ')}]` : 'missing'}.`
    );
  }

  const device = getDevice();
  if (!device) {
    throw new Error('[SoftEmbedding] GPU device not available.');
  }

  const outputBytes = numTokens * hiddenSize * Float32Array.BYTES_PER_ELEMENT;
  const output = options.outputBuffer ?? acquireBuffer(outputBytes, undefined, 'soft_embedding_split_output');
  const ownsOutput = options.outputBuffer ? null : output;
  const pipeline = await createPipeline('soft_embedding', 'split_f16');

  try {
    for (let sectionIndex = 0; sectionIndex < splitEmbedding.sections.length; sectionIndex += 1) {
      await dispatchSplitSection({
        device,
        pipeline,
        softmaxTensor,
        section: splitEmbedding.sections[sectionIndex],
        sectionIndex,
        output,
        numTokens,
        hiddenSize,
        vocabSize,
      });
    }
  } catch (error) {
    if (ownsOutput) {
      releaseBuffer(ownsOutput);
    }
    throw error;
  }

  return createTensor(output, 'f32', [numTokens, hiddenSize], 'soft_embedding_split_output');
}
