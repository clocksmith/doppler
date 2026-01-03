
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor, inferOutputDtype, dtypeBytes } from '../tensor.js';
import { WORKGROUP_SIZES, VEC4_ELEMENTS_PER_WG } from './constants.js';
import { unifiedKernelWrapper } from './utils.js';
import { castF16ToF32, castF32ToF16, recordCastF16ToF32, recordCastF32ToF16 } from './cast.js';
import { selectRuleValue } from './rule-registry.js';


function selectResidualVariant(outputDtype, useVec4) {
  return selectRuleValue(
    'residual',
    'residualVariant',
    { outputDtype, useVec4 }
  );
}

function selectBiasAddVariant(dataDtype, biasDtype) {
  return selectRuleValue(
    'residual',
    'biasAddVariant',
    { dataDtype, biasDtype }
  );
}

async function alignResidualInputs(a, b, recorder) {
  if (a.dtype === b.dtype) {
    return { a, b, temps: [] };
  }
  if (a.dtype === 'f16' && b.dtype === 'f32') {
    const casted = recorder ? await recordCastF16ToF32(recorder, a) : await castF16ToF32(a);
    return { a: casted, b, temps: [casted.buffer] };
  }
  if (a.dtype === 'f32' && b.dtype === 'f16') {
    const casted = recorder ? await recordCastF16ToF32(recorder, b) : await castF16ToF32(b);
    return { a, b: casted, temps: [casted.buffer] };
  }
  return { a, b, temps: [] };
}

async function alignBiasTensor(data, bias, recorder) {
  if (data.dtype === bias.dtype) {
    return { bias, temps: [] };
  }
  if (data.dtype === 'f16' && bias.dtype === 'f32') {
    const casted = recorder ? await recordCastF32ToF16(recorder, bias) : await castF32ToF16(bias);
    return { bias: casted, temps: [casted.buffer] };
  }
  if (data.dtype === 'f32' && bias.dtype === 'f16') {
    const casted = recorder ? await recordCastF16ToF32(recorder, bias) : await castF16ToF32(bias);
    return { bias: casted, temps: [casted.buffer] };
  }
  return { bias, temps: [] };
}

function cleanupTemps(temps, recorder) {
  for (const temp of temps) {
    if (recorder) {
      recorder.trackTemporaryBuffer(temp);
    } else {
      releaseBuffer(temp);
    }
  }
}

async function _residualAdd(target, a, b, size, options = {}) {
  const recorder = target && typeof target.beginComputePass === 'function' ? target : null;
  const { useVec4 = true, outputBuffer = null } = options;

  const { a: aAligned, b: bAligned, temps } = await alignResidualInputs(a, b, recorder);
  const outputDtype = inferOutputDtype(aAligned, bAligned);
  const bytesPerElement = dtypeBytes(outputDtype);

  const variant = selectResidualVariant(outputDtype, useVec4);
  const outputSize = size * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'residual_output');

  const workgroups = useVec4
    ? Math.ceil(size / VEC4_ELEMENTS_PER_WG)
    : Math.ceil(size / WORKGROUP_SIZES.DEFAULT);

  await unifiedKernelWrapper(
    'residual', target, variant,
    [aAligned, bAligned, output],
    { size },
    workgroups
  );

  cleanupTemps(temps, recorder);
  return createTensor(output, outputDtype, [size], 'residual_output');
}

async function _biasAdd(target, data, bias, numTokens, dim, options = {}) {
  const recorder = target && typeof target.beginComputePass === 'function' ? target : null;
  const { dataOffset = 0, biasOffset = 0 } = options;

  const { bias: biasAligned, temps } = await alignBiasTensor(data, bias, recorder);
  const variant = selectBiasAddVariant(data.dtype, biasAligned.dtype);

  const workgroups = Math.ceil((numTokens * dim) / WORKGROUP_SIZES.DEFAULT);

  await unifiedKernelWrapper(
    'bias_add', target, variant,
    [data, biasAligned],
    { num_tokens: numTokens, dim, data_offset: dataOffset, bias_offset: biasOffset },
    workgroups
  );

  cleanupTemps(temps, recorder);
  return createTensor(data.buffer, data.dtype, [numTokens, dim], 'bias_add_output');
}

export async function runResidualAdd(a, b, size, options = {}) {
  return _residualAdd(null, a, b, size, options);
}

export async function recordResidualAdd(recorder, a, b, size, options = {}) {
  return _residualAdd(recorder, a, b, size, options);
}

export async function runBiasAdd(data, bias, numTokens, dim, options = {}) {
  return _biasAdd(null, data, bias, numTokens, dim, options);
}

export async function recordBiasAdd(recorder, data, bias, numTokens, dim, options = {}) {
  return _biasAdd(recorder, data, bias, numTokens, dim, options);
}
