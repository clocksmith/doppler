
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { unifiedKernelWrapper } from './utils.js';
import { castF16ToF32, recordCastF16ToF32 } from './cast.js';

function resolveDimensions(softmax, options) {
  const inferred = softmax.shape.length >= 2 ? softmax.shape : [];
  const numTokens = options.numTokens ?? inferred[0];
  const vocabSize = options.vocabSize ?? inferred[1];
  if (!numTokens || !vocabSize) {
    throw new Error('cross entropy loss requires numTokens and vocabSize');
  }
  return { numTokens, vocabSize };
}

async function _crossEntropyLoss(target, softmax, targets, options = {}) {
  const recorder = target && typeof target.beginComputePass === 'function' ? target : null;
  const { outputBuffer = null } = options;
  const { numTokens, vocabSize } = resolveDimensions(softmax, options);

  const inputTensor = softmax.dtype === 'f16'
    ? (recorder ? await recordCastF16ToF32(recorder, softmax) : await castF16ToF32(softmax))
    : softmax;

  const outputSize = numTokens * 4;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'cross_entropy_loss_output');

  await unifiedKernelWrapper(
    'cross_entropy_loss', target, 'default',
    [inputTensor, targets, outputBuf],
    { num_tokens: numTokens, vocab_size: vocabSize },
    Math.ceil(numTokens / WORKGROUP_SIZES.DEFAULT)
  );

  if (inputTensor !== softmax) {
    if (recorder) {
      recorder.trackTemporaryBuffer(inputTensor.buffer);
    } else {
      releaseBuffer(inputTensor.buffer);
    }
  }

  return createTensor(outputBuf, 'f32', [numTokens], 'cross_entropy_loss_output');
}

export async function runCrossEntropyLoss(softmax, targets, options = {}) {
  return _crossEntropyLoss(null, softmax, targets, options);
}

export async function recordCrossEntropyLoss(recorder, softmax, targets, options = {}) {
  return _crossEntropyLoss(recorder, softmax, targets, options);
}
