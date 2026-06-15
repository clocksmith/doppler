import { getDevice } from '../device.js';
import { createTensor } from '../tensor.js';
import { isSplitWeightBuffer, isWeightBuffer, getWeightDtype, getLayout } from '../weight-buffer.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createPipeline, createUniformBufferWithView, createBindGroupWithValidation } from './utils.js';
import { dispatchKernel } from './dispatch.js';
import { runMatmul } from './matmul.js';
import { runResidualAdd } from './residual.js';

const UNIFORM_SIZE = 32;
const WORKGROUP_X = 8;
const WORKGROUP_Y = 8;
const LOGITS_WORKGROUP_SIZE = 256;
const DEFAULT_LOGITS_CHUNK_ROWS = 32768;
const STORAGE_OFFSET_ALIGNMENT = 256;

function validatePositiveInt(value, label) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`[SoftEmbedding] ${label} must be a positive integer; got ${String(value)}.`);
  }
}

function validatePositiveNumber(value, label) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    throw new Error(`[SoftEmbedding] ${label} must be a positive finite number; got ${String(value)}.`);
  }
}

function gcd(a, b) {
  let x = Math.abs(a);
  let y = Math.abs(b);
  while (y !== 0) {
    const next = x % y;
    x = y;
    y = next;
  }
  return x;
}

function alignChunkRowsForWeightOffset(chunkRows, hiddenSize, vocabSize) {
  if (chunkRows >= vocabSize) {
    return vocabSize;
  }
  const rowBytes = hiddenSize * Uint16Array.BYTES_PER_ELEMENT;
  const alignmentRows = STORAGE_OFFSET_ALIGNMENT / gcd(rowBytes, STORAGE_OFFSET_ALIGNMENT);
  const alignedRows = Math.floor(chunkRows / alignmentRows) * alignmentRows;
  return Math.min(vocabSize, Math.max(alignmentRows, alignedRows));
}

function resolveLogitsChunkRows(device, numTokens, hiddenSize, vocabSize, requestedRows) {
  const requested = Number.isInteger(requestedRows) && requestedRows > 0
    ? requestedRows
    : DEFAULT_LOGITS_CHUNK_ROWS;
  const maxBinding = Math.min(
    device.limits?.maxStorageBufferBindingSize ?? Number.POSITIVE_INFINITY,
    device.limits?.maxBufferSize ?? Number.POSITIVE_INFINITY
  );
  const maxByProbability = Number.isFinite(maxBinding)
    ? Math.floor(maxBinding / (numTokens * Float32Array.BYTES_PER_ELEMENT))
    : requested;
  const maxByEmbedding = Number.isFinite(maxBinding)
    ? Math.floor(maxBinding / (hiddenSize * Uint16Array.BYTES_PER_ELEMENT))
    : requested;
  const bounded = Math.max(1, Math.min(requested, maxByProbability, maxByEmbedding, vocabSize));
  return alignChunkRowsForWeightOffset(bounded, hiddenSize, vocabSize);
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

function createLogitsUniformBuffer(device, numTokens, hiddenSize, vocabSize, rowStart, rowCount, temperature) {
  return createUniformBufferWithView(
    'soft_embedding_logits_uniforms',
    UNIFORM_SIZE,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, hiddenSize, true);
      view.setUint32(8, vocabSize, true);
      view.setUint32(12, rowStart, true);
      view.setUint32(16, rowCount, true);
      view.setFloat32(20, temperature, true);
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

async function dispatchLogitsNormStats({
  device,
  logitsTensor,
  rowMaxBuffer,
  rowSumBuffer,
  numTokens,
  hiddenSize,
  vocabSize,
  temperature,
}) {
  const pipeline = await createPipeline('soft_embedding', 'logits_norm_stats');
  const uniformBuffer = createLogitsUniformBuffer(
    device,
    numTokens,
    hiddenSize,
    vocabSize,
    0,
    vocabSize,
    temperature
  );
  try {
    const bindGroup = await createBindGroupWithValidation(device, {
      label: 'soft_embedding_logits_norm_stats_bind_group',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: logitsTensor.buffer } },
        { binding: 2, resource: { buffer: rowMaxBuffer } },
        { binding: 3, resource: { buffer: rowSumBuffer } },
      ],
    }, 'soft_embedding/logits_norm_stats');
    dispatchKernel(null, pipeline, bindGroup, numTokens, 'soft_embedding_logits_norm_stats');
  } finally {
    uniformBuffer.destroy();
  }
}

async function dispatchProbabilityChunk({
  device,
  logitsTensor,
  rowMaxBuffer,
  rowSumBuffer,
  probabilityBuffer,
  numTokens,
  hiddenSize,
  vocabSize,
  rowStart,
  rowCount,
  temperature,
}) {
  const pipeline = await createPipeline('soft_embedding', 'logits_probability_chunk');
  const uniformBuffer = createLogitsUniformBuffer(
    device,
    numTokens,
    hiddenSize,
    vocabSize,
    rowStart,
    rowCount,
    temperature
  );
  try {
    const bindGroup = await createBindGroupWithValidation(device, {
      label: 'soft_embedding_logits_probability_chunk_bind_group',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: logitsTensor.buffer } },
        { binding: 2, resource: { buffer: rowMaxBuffer } },
        { binding: 3, resource: { buffer: rowSumBuffer } },
        { binding: 4, resource: { buffer: probabilityBuffer } },
      ],
    }, 'soft_embedding/logits_probability_chunk');
    dispatchKernel(
      null,
      pipeline,
      bindGroup,
      [Math.ceil(rowCount / LOGITS_WORKGROUP_SIZE), numTokens, 1],
      'soft_embedding_logits_probability_chunk'
    );
  } finally {
    uniformBuffer.destroy();
  }
}

export async function runSoftEmbeddingLogitsF16(logitsTensor, embedding, numTokens, hiddenSize, vocabSize, options = {}) {
  validatePositiveInt(numTokens, 'numTokens');
  validatePositiveInt(hiddenSize, 'hiddenSize');
  validatePositiveInt(vocabSize, 'vocabSize');
  const temperature = options.temperature ?? 1.0;
  validatePositiveNumber(temperature, 'temperature');

  if (logitsTensor?.dtype !== 'f32') {
    throw new Error(`[SoftEmbedding] logits f16 path requires f32 logits input, got "${logitsTensor?.dtype ?? 'missing'}".`);
  }
  if (!isWeightBuffer(embedding)) {
    throw new Error('[SoftEmbedding] logits f16 path requires a WeightBuffer embedding table.');
  }
  if (getWeightDtype(embedding) !== 'f16' || getLayout(embedding) !== 'row') {
    throw new Error(
      `[SoftEmbedding] logits path supports row-major f16 embeddings only; ` +
      `got dtype=${getWeightDtype(embedding)}, layout=${getLayout(embedding)}.`
    );
  }
  const shape = Array.isArray(embedding.shape) ? embedding.shape : null;
  if (!shape || shape.length !== 2 || shape[0] !== vocabSize || shape[1] !== hiddenSize) {
    throw new Error(
      `[SoftEmbedding] embedding shape mismatch: expected [${vocabSize}, ${hiddenSize}], ` +
      `got ${shape ? `[${shape.join(', ')}]` : 'missing'}.`
    );
  }

  const device = getDevice();
  if (!device) {
    throw new Error('[SoftEmbedding] GPU device not available.');
  }

  const statsBytes = numTokens * Float32Array.BYTES_PER_ELEMENT;
  const rowMaxBuffer = acquireBuffer(statsBytes, undefined, 'soft_embedding_logits_row_max');
  const rowSumBuffer = acquireBuffer(statsBytes, undefined, 'soft_embedding_logits_row_sum');
  let output = null;
  let completed = false;

  try {
    await dispatchLogitsNormStats({
      device,
      logitsTensor,
      rowMaxBuffer,
      rowSumBuffer,
      numTokens,
      hiddenSize,
      vocabSize,
      temperature,
    });

    const chunkRows = resolveLogitsChunkRows(
      device,
      numTokens,
      hiddenSize,
      vocabSize,
      options.chunkRows
    );
    for (let rowStart = 0; rowStart < vocabSize; rowStart += chunkRows) {
      const rowCount = Math.min(chunkRows, vocabSize - rowStart);
      const probabilityBytes = numTokens * rowCount * Float32Array.BYTES_PER_ELEMENT;
      const probabilityBuffer = acquireBuffer(probabilityBytes, undefined, 'soft_embedding_logits_probability_chunk');
      let chunkOutput = null;
      try {
        await dispatchProbabilityChunk({
          device,
          logitsTensor,
          rowMaxBuffer,
          rowSumBuffer,
          probabilityBuffer,
          numTokens,
          hiddenSize,
          vocabSize,
          rowStart,
          rowCount,
          temperature,
        });
        const probabilityTensor = createTensor(
          probabilityBuffer,
          'f32',
          [numTokens, rowCount],
          'soft_embedding_logits_probability_chunk'
        );
        chunkOutput = await runMatmul(
          probabilityTensor,
          embedding,
          numTokens,
          hiddenSize,
          rowCount,
          {
            bOffset: rowStart * hiddenSize * Uint16Array.BYTES_PER_ELEMENT,
            transposeB: false,
            role: 'diffusion_gemma_self_conditioning_embed',
            outputDtype: 'f32',
          }
        );
        if (!output) {
          output = chunkOutput;
          chunkOutput = null;
        } else {
          const nextOutput = await runResidualAdd(output, chunkOutput, numTokens * hiddenSize, {
            useVec4: true,
          });
          releaseBuffer(output.buffer);
          releaseBuffer(chunkOutput.buffer);
          output = nextOutput;
          chunkOutput = null;
        }
      } finally {
        releaseBuffer(probabilityBuffer);
        if (chunkOutput?.buffer) {
          releaseBuffer(chunkOutput.buffer);
        }
      }
    }

    completed = true;
    return createTensor(output.buffer, 'f32', [numTokens, hiddenSize], 'soft_embedding_logits_output');
  } finally {
    releaseBuffer(rowMaxBuffer);
    releaseBuffer(rowSumBuffer);
    if (!completed && output?.buffer) {
      releaseBuffer(output.buffer);
    }
  }
}
