
import { runSoftmax, runCrossEntropyLoss, castF16ToF32 } from '../gpu/kernels/index.js';
import { releaseBuffer } from '../gpu/buffer-pool.js';
import { OpType } from './autograd.js';

export async function crossEntropyLoss(logits, targets, config, tape) {
  if (!tape || typeof tape.record !== 'function') {
    throw new Error('crossEntropyLoss requires an autograd tape');
  }

  if (logits.shape.length < 2) {
    throw new Error('crossEntropyLoss expects logits with shape [numTokens, vocabSize]');
  }

  const numTokens = logits.shape[0];
  const vocabSize = logits.shape[1];
  const totalTargets = targets.shape.reduce((acc, value) => acc * value, 1);
  if (totalTargets !== numTokens) {
    throw new Error(`crossEntropyLoss targets size ${totalTargets} does not match logits rows ${numTokens}`);
  }

  const logitsF32 = logits.dtype === 'f16' ? await castF16ToF32(logits) : logits;

  const softmax = await tape.record(
    OpType.SOFTMAX,
    (input) => runSoftmax(input, -1, { batchSize: numTokens, size: vocabSize }),
    [logitsF32],
    { rows: numTokens, cols: vocabSize }
  );

  if (logitsF32 !== logits) {
    releaseBuffer(logitsF32.buffer);
  }

  return tape.record(
    OpType.CROSS_ENTROPY,
    (input, target) => runCrossEntropyLoss(input, target, { numTokens, vocabSize }),
    [softmax, targets],
    { numTokens, vocabSize }
  );
}
