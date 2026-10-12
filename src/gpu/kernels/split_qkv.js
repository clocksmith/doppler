import { executePairedSplitQKV } from './paired-split.js';

import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { planSplitQKVDispatch } from './split-dispatch.js';
import { unifiedKernelWrapper } from './kernel-execution.js';
import { selectRuleValue } from './rule-registry.js';

export { planSplitQKVDispatch } from './split-dispatch.js';

async function _splitQKV(target, qkvTensor, options) {
  if (qkvTensor.precision !== undefined || qkvTensor.components !== undefined || qkvTensor.elementStrideBytes !== undefined) {
    return executePairedSplitQKV(target, qkvTensor, options);
  }
  const { numTokens, qSize, kSize, vSize, qTensor = null, kTensor = null, vTensor = null } = options;
  const ownsQ = qTensor == null;
  const ownsK = kTensor == null;
  const ownsV = vTensor == null;

  const outputDtype = qkvTensor.dtype;
  const pipelineVariant = selectRuleValue('splitQkv', 'variant', { outputDtype });
  const bytesPerElement = dtypeBytes(outputDtype);

  const qBuffer = qTensor?.buffer || acquireBuffer(numTokens * qSize * bytesPerElement, undefined, 'Q');
  const kBuffer = kTensor?.buffer || acquireBuffer(numTokens * kSize * bytesPerElement, undefined, 'K');
  const vBuffer = vTensor?.buffer || acquireBuffer(numTokens * vSize * bytesPerElement, undefined, 'V');

  const totalElements = numTokens * (qSize + kSize + vSize);

  try {
    await unifiedKernelWrapper(
      'split_qkv', target, pipelineVariant,
      [qkvTensor, qBuffer, kBuffer, vBuffer],
      { numTokens, qSize, kSize, vSize },
      planSplitQKVDispatch(totalElements)
    );

    const Q = qTensor || createTensor(qBuffer, outputDtype, [numTokens, qSize], 'Q');
    const K = kTensor || createTensor(kBuffer, outputDtype, [numTokens, kSize], 'K');
    const V = vTensor || createTensor(vBuffer, outputDtype, [numTokens, vSize], 'V');

    return { Q, K, V };
  } catch (error) {
    if (ownsQ) releaseBuffer(qBuffer);
    if (ownsK) releaseBuffer(kBuffer);
    if (ownsV) releaseBuffer(vBuffer);
    throw error;
  }
}

export async function runSplitQKV(qkvTensor, options) {
  return _splitQKV(null, qkvTensor, options);
}

export async function recordSplitQKV(recorder, qkvTensor, options) {
  return _splitQKV(recorder, qkvTensor, options);
}
