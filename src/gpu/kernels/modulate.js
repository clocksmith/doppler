import { acquireBuffer } from '../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { unifiedKernelWrapper } from './utils.js';
import { selectRuleValue } from './rule-registry.js';

function selectModulateVariant(inputDtype, modDtype) {
  return selectRuleValue('modulate', 'variant', { inputDtype, modDtype });
}

async function _modulate(target, input, mod, options = {}) {
  const {
    numTokens, hiddenSize,
    scaleOffset = 0, shiftOffset = 0, gateOffset = 0,
    hasGate = false, addOne = true,
    outputBuffer = null,
  } = options;

  if (!Number.isFinite(numTokens) || !Number.isFinite(hiddenSize)) {
    throw new Error('Modulate requires numTokens and hiddenSize.');
  }

  const variant = selectModulateVariant(input.dtype, mod.dtype);
  const bytesPerElement = dtypeBytes(input.dtype);
  const outputSize = numTokens * hiddenSize * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'modulate_output');

  await unifiedKernelWrapper(
    'modulate', target, variant,
    [input, mod, output],
    {
      num_tokens: numTokens, hidden_size: hiddenSize,
      scale_offset: scaleOffset, shift_offset: shiftOffset,
      gate_offset: gateOffset, has_gate: hasGate ? 1 : 0,
      add_one: addOne ? 1 : 0, _pad0: 0,
    },
    Math.ceil((numTokens * hiddenSize) / 256)
  );

  return createTensor(output, input.dtype, [numTokens, hiddenSize], 'modulate_output');
}

export async function runModulate(input, mod, options = {}) {
  return _modulate(null, input, mod, options);
}

export async function recordModulate(recorder, input, mod, options = {}) {
  return _modulate(recorder, input, mod, options);
}
