
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { unifiedKernelWrapper } from './utils.js';
import { selectRuleValue } from './rule-registry.js';

async function _splitQG(target, qgTensor, options) {
  const { numTokens, numHeads, headDim, qTensor = null, gTensor = null } = options;
  const ownsQ = qTensor == null;
  const ownsG = gTensor == null;

  const outputDtype = qgTensor.dtype;
  const pipelineVariant = selectRuleValue('splitQg', 'variant', { outputDtype });
  const bytesPerElement = dtypeBytes(outputDtype);
  const qSize = numHeads * headDim;

  const qBuffer = qTensor?.buffer || acquireBuffer(numTokens * qSize * bytesPerElement, undefined, 'Q');
  const gBuffer = gTensor?.buffer || acquireBuffer(numTokens * qSize * bytesPerElement, undefined, 'Q_gate');

  try {
    await unifiedKernelWrapper(
      'split_qg', target, pipelineVariant,
      [qgTensor, qBuffer, gBuffer],
      { num_tokens: numTokens, num_heads: numHeads, head_dim: headDim, _pad: 0 },
      Math.ceil((numTokens * qSize) / WORKGROUP_SIZES.DEFAULT)
    );

    const Q = qTensor || createTensor(qBuffer, outputDtype, [numTokens, qSize], 'Q');
    const G = gTensor || createTensor(gBuffer, outputDtype, [numTokens, qSize], 'Q_gate');

    return { Q, G };
  } catch (error) {
    if (ownsQ) releaseBuffer(qBuffer);
    if (ownsG) releaseBuffer(gBuffer);
    throw error;
  }
}

export async function runSplitQG(qgTensor, options) {
  return _splitQG(null, qgTensor, options);
}

export async function recordSplitQG(recorder, qgTensor, options) {
  return _splitQG(recorder, qgTensor, options);
}
