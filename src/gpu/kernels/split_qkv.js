
import { acquireBuffer } from '../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { unifiedKernelWrapper } from './utils.js';
import { selectRuleValue } from './rule-registry.js';

async function _splitQKV(target, qkvTensor, options) {
  const { numTokens, qSize, kSize, vSize, qTensor = null, kTensor = null, vTensor = null } = options;

  const outputDtype = qkvTensor.dtype;
  const pipelineVariant = selectRuleValue('splitQkv', 'variant', { outputDtype });
  const bytesPerElement = dtypeBytes(outputDtype);

  const qBuffer = qTensor?.buffer || acquireBuffer(numTokens * qSize * bytesPerElement, undefined, 'Q');
  const kBuffer = kTensor?.buffer || acquireBuffer(numTokens * kSize * bytesPerElement, undefined, 'K');
  const vBuffer = vTensor?.buffer || acquireBuffer(numTokens * vSize * bytesPerElement, undefined, 'V');

  const totalElements = numTokens * (qSize + kSize + vSize);

  await unifiedKernelWrapper(
    'split_qkv', target, pipelineVariant,
    [qkvTensor, qBuffer, kBuffer, vBuffer],
    { num_tokens: numTokens, q_size: qSize, k_size: kSize, v_size: vSize },
    Math.ceil(totalElements / WORKGROUP_SIZES.DEFAULT)
  );

  const Q = qTensor || createTensor(qBuffer, outputDtype, [numTokens, qSize], 'Q');
  const K = kTensor || createTensor(kBuffer, outputDtype, [numTokens, kSize], 'K');
  const V = vTensor || createTensor(vBuffer, outputDtype, [numTokens, vSize], 'V');

  return { Q, K, V };
}

export async function runSplitQKV(qkvTensor, options) {
  return _splitQKV(null, qkvTensor, options);
}

export async function recordSplitQKV(recorder, qkvTensor, options) {
  return _splitQKV(recorder, qkvTensor, options);
}
