import { createTensor } from '../tensor.js';
import { unifiedKernelWrapper, withKernelOutput } from './kernel-execution.js';
import { planSplitQKVDispatch } from './split-dispatch.js';
import { selectRuleValue } from './rule-registry.js';

/** @type {Readonly<import('../tensor.js').TensorRepresentation>} */
const PAIR = Object.freeze({ precision: 'two-f32', components: 2, elementStrideBytes: 8 });
/** @param {number} value */
function positive(value) {
  if (!Number.isSafeInteger(value) || value <= 0 || value > 0xffffffff) throw new Error('[PairedSplit] Geometry must be positive U32.');
}
/** @param {import('../tensor.js').Tensor} input @param {number} rows @param {number} width */
function validate(input, rows, width) {
  positive(rows); positive(width); positive(rows * width);
  if (input.dtype !== 'f32' || input.precision !== 'two-f32' || input.components !== 2 || input.elementStrideBytes !== 8) {
    throw new Error('[PairedSplit] Explicit interleaved two-F32 tensor required.');
  }
  if (input.shape.length !== 2 || input.shape[0] !== rows || input.shape[1] !== width || input.buffer.size < rows * width * 8) {
    throw new Error('[PairedSplit] Exact logical shape and complete paired storage required.');
  }
}
/** @param {import('../tensor.js').Tensor} input @param {(import('../tensor.js').Tensor | null | undefined)[]} outputs @param {number[]} widths @param {number} rows */
function validateOutputs(input, outputs, widths, rows) {
  const buffers = new Set([input.buffer]);
  outputs.forEach((output, index) => {
    if (!output) return;
    validate(output, rows, widths[index]);
    if (buffers.has(output.buffer)) throw new Error('[PairedSplit] Outputs may not alias inputs or each other.');
    buffers.add(output.buffer);
  });
}
/** @type {typeof import('./paired-split.js').executePairedSplitQG} */
export async function executePairedSplitQG(target, input, options) {
  const { numTokens, numHeads, headDim, qTensor = null, gTensor = null } = options;
  positive(numHeads); positive(headDim);
  const width = numHeads * headDim;
  validate(input, numTokens, width * 2);
  validateOutputs(input, [qTensor, gTensor], [width, width], numTokens);
  const bytes = numTokens * width * 8;
  return withKernelOutput(target, qTensor?.buffer ?? null, bytes, 'paired_split_q', async q =>
    withKernelOutput(target, gTensor?.buffer ?? null, bytes, 'paired_split_gate', async g => {
      const variant = selectRuleValue('splitQg', 'variant', { precision: input.precision, outputDtype: input.dtype });
      await unifiedKernelWrapper('split_qg', target, variant, [input, q, g], {
        num_tokens: numTokens, num_heads: numHeads, head_dim: headDim,
      }, planSplitQKVDispatch(numTokens * width));
      return { Q: qTensor ?? createTensor(q, 'f32', [numTokens, width], 'Q', PAIR),
        G: gTensor ?? createTensor(g, 'f32', [numTokens, width], 'Q_gate', PAIR) };
    }));
}
/** @type {typeof import('./paired-split.js').executePairedSplitQKV} */
export async function executePairedSplitQKV(target, input, options) {
  const { numTokens, qSize, kSize, vSize, qTensor = null, kTensor = null, vTensor = null } = options;
  positive(qSize); positive(kSize); positive(vSize);
  validate(input, numTokens, qSize + kSize + vSize);
  validateOutputs(input, [qTensor, kTensor, vTensor], [qSize, kSize, vSize], numTokens);
  return withKernelOutput(target, qTensor?.buffer ?? null, numTokens * qSize * 8, 'paired_split_q', async q =>
    withKernelOutput(target, kTensor?.buffer ?? null, numTokens * kSize * 8, 'paired_split_k', async k =>
      withKernelOutput(target, vTensor?.buffer ?? null, numTokens * vSize * 8, 'paired_split_v', async v => {
        const variant = selectRuleValue('splitQkv', 'variant', { precision: input.precision, outputDtype: input.dtype });
        await unifiedKernelWrapper('split_qkv', target, variant, [input, q, k, v], {
          numTokens, qSize, kSize, vSize,
        }, planSplitQKVDispatch(numTokens * (qSize + kSize + vSize)));
        return { Q: qTensor ?? createTensor(q, 'f32', [numTokens, qSize], 'Q', PAIR),
          K: kTensor ?? createTensor(k, 'f32', [numTokens, kSize], 'K', PAIR),
          V: vTensor ?? createTensor(v, 'f32', [numTokens, vSize], 'V', PAIR) };
      })));
}
