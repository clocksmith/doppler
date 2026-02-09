
import { acquireBuffer } from '../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { unifiedKernelWrapper } from './utils.js';
import { selectRuleValue } from './rule-registry.js';

function selectGeluVariant(isF16) {
  return selectRuleValue('gelu', 'variant', { isF16 });
}

function resolveOverrides(context) {
  const overrides = selectRuleValue('gelu', 'overrides', context);
  return overrides && Object.keys(overrides).length > 0 ? overrides : null;
}

async function _gelu(target, input, options = {}) {
  const { size, gate = null, outputBuffer = null } = options;

  const isF16 = input.dtype === 'f16';
  const bytesPerElement = dtypeBytes(input.dtype);
  const variant = selectGeluVariant(isF16);
  const overrides = resolveOverrides({ hasGate: Boolean(gate), useRowsplit: false });

  const inferredSize = size || (input.buffer.size / bytesPerElement);
  const outputSize = inferredSize * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'gelu_output');
  const gateBuffer = gate ?? input;

  await unifiedKernelWrapper(
    'gelu', target, variant,
    [input, output, gateBuffer],
    { size: inferredSize, rowsplit_dim: 0 },
    Math.ceil(inferredSize / WORKGROUP_SIZES.DEFAULT),
    overrides
  );

  return createTensor(output, input.dtype, [inferredSize], 'gelu_output');
}

export async function runGeLU(input, options = {}) {
  return _gelu(null, input, options);
}

export async function recordGeLU(recorder, input, options = {}) {
  return _gelu(recorder, input, options);
}
