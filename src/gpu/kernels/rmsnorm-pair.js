import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { padToQ4KBlock } from '../../config/schema/index.js';
import { createTensor } from '../tensor.js';
import { getBuffer } from '../weight-buffer.js';
import { getKernelCapabilities } from '../device.js';
import { unifiedKernelWrapper } from './utils.js';
import {
  assertRMSNormWeightBuffer,
  planRMSNormDispatch,
  resolveNormWeightDtype,
} from './rmsnorm.js';

export const RMSNORM_PAIR_CACHE_LIMIT = 4608;

function inferHiddenSize(input, hiddenSize) {
  if (hiddenSize != null) return hiddenSize;
  const shape = input?.shape;
  if (Array.isArray(shape) && shape.length > 0) {
    return shape[shape.length - 1];
  }
  return null;
}

function resolveVariant() {
  const caps = getKernelCapabilities();
  return caps?.hasSubgroups ? 'subgroup' : 'default';
}

function resolveResidualVariant() {
  const caps = getKernelCapabilities();
  if (caps?.hasSubgroups === true) {
    return 'subgroup';
  }
  throw new Error('residual_rmsnorm_pair requires subgroup support; disable usePostFfnNextInputRMSNormPairFusion for this adapter.');
}

function resolveDispatchLabel(label) {
  if (typeof label !== 'string' || label.length === 0) {
    return 'rmsnorm_pair';
  }
  const normalized = label.replace(/^L\d+\./, '').replace(/\s+/g, '_');
  return `rmsnorm_pair:${normalized}`;
}

function resolveResidualDispatchLabel(label) {
  if (typeof label !== 'string' || label.length === 0) {
    return 'residual_rmsnorm_pair';
  }
  const normalized = label.replace(/^L\d+\./, '').replace(/\s+/g, '_');
  return `residual_rmsnorm_pair:${normalized}`;
}

function assertRMSNormPairInputs(input, residual, hiddenSize) {
  if (input?.dtype !== 'f32') {
    throw new Error(`[rmsnorm_pair] input dtype must be f32; got "${String(input?.dtype)}".`);
  }
  if (residual && residual.dtype !== 'f32') {
    throw new Error(`[rmsnorm_pair] residual dtype must be f32; got "${String(residual.dtype)}".`);
  }
  if (!Number.isInteger(hiddenSize) || hiddenSize <= 0) {
    throw new Error(`[rmsnorm_pair] hiddenSize must be a positive integer; got "${String(hiddenSize)}".`);
  }
  if (hiddenSize > RMSNORM_PAIR_CACHE_LIMIT) {
    throw new Error(
      `[rmsnorm_pair] hiddenSize=${hiddenSize} exceeds cache limit ${RMSNORM_PAIR_CACHE_LIMIT}.`
    );
  }
}

async function runRMSNormPairImpl(
  target,
  input,
  residual,
  postWeight,
  preWeight,
  eps,
  options = {}
) {
  const {
    batchSize = 1,
    hiddenSize = null,
    rmsNormWeightOffset = false,
    label = null,
    postOutputBuffer = null,
    preOutputBuffer = null,
  } = options;
  const inferredHiddenSize = inferHiddenSize(input, hiddenSize);
  assertRMSNormPairInputs(input, residual, inferredHiddenSize);

  const postWeightBuffer = getBuffer(postWeight);
  const preWeightBuffer = getBuffer(preWeight);
  assertRMSNormWeightBuffer(postWeight, postWeightBuffer, inferredHiddenSize);
  assertRMSNormWeightBuffer(preWeight, preWeightBuffer, inferredHiddenSize);
  const postWeightDtype = resolveNormWeightDtype(postWeight, inferredHiddenSize);
  const preWeightDtype = resolveNormWeightDtype(preWeight, inferredHiddenSize);

  const paddedHiddenSize = padToQ4KBlock(inferredHiddenSize);
  const outputSize = batchSize * paddedHiddenSize * 4;
  const ownsPostOutput = postOutputBuffer == null;
  const ownsPreOutput = preOutputBuffer == null;
  const postOutput = postOutputBuffer ?? acquireBuffer(outputSize, undefined, 'rmsnorm_pair_post_output');
  const preOutput = preOutputBuffer ?? acquireBuffer(outputSize, undefined, 'rmsnorm_pair_pre_output');
  const residualBuffer = residual?.buffer || residual || input.buffer;
  const dispatchPlan = planRMSNormDispatch(target, batchSize);
  const variant = resolveVariant();
  let completed = false;

  try {
    await unifiedKernelWrapper(
      'rmsnorm_pair',
      target,
      variant,
      [input, residualBuffer, postWeightBuffer, preWeightBuffer, postOutput, preOutput],
      {
        hidden_size: inferredHiddenSize,
        num_tokens: batchSize,
        eps,
        has_residual: residual ? 1 : 0,
        token_stride: dispatchPlan.tokenStride,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
      },
      dispatchPlan.workgroups,
      {
        RMS_NORM_OFFSET: rmsNormWeightOffset,
        POST_WEIGHT_IS_F16: postWeightDtype === 'f16',
        PRE_WEIGHT_IS_F16: preWeightDtype === 'f16',
      },
      null,
      resolveDispatchLabel(label)
    );

    completed = true;
    return {
      postAttn: createTensor(postOutput, 'f32', [batchSize, inferredHiddenSize], 'rmsnorm_pair_post_output'),
      ffnInput: createTensor(preOutput, 'f32', [batchSize, inferredHiddenSize], 'rmsnorm_pair_pre_output'),
    };
  } finally {
    if (!completed) {
      if (ownsPostOutput) releaseBuffer(postOutput);
      if (ownsPreOutput) releaseBuffer(preOutput);
    }
  }
}

export async function runSandwichRMSNormPair(
  input,
  residual,
  postWeight,
  preWeight,
  eps,
  options = {}
) {
  return runRMSNormPairImpl(null, input, residual, postWeight, preWeight, eps, options);
}

export async function recordSandwichRMSNormPair(
  recorder,
  input,
  residual,
  postWeight,
  preWeight,
  eps,
  options = {}
) {
  return runRMSNormPairImpl(recorder, input, residual, postWeight, preWeight, eps, options);
}

async function runResidualRMSNormPairImpl(
  target,
  input,
  residual,
  normWeight,
  eps,
  options = {}
) {
  const {
    batchSize = 1,
    hiddenSize = null,
    rmsNormWeightOffset = false,
    label = null,
    residualOutputBuffer = null,
    normOutputBuffer = null,
    outputScale = 1,
  } = options;
  const inferredHiddenSize = inferHiddenSize(input, hiddenSize);
  assertRMSNormPairInputs(input, residual, inferredHiddenSize);

  const normWeightBuffer = getBuffer(normWeight);
  assertRMSNormWeightBuffer(normWeight, normWeightBuffer, inferredHiddenSize);
  const normWeightDtype = resolveNormWeightDtype(normWeight, inferredHiddenSize);
  const residualScale = Number(outputScale);
  if (!Number.isFinite(residualScale)) {
    throw new Error(`[residual_rmsnorm_pair] outputScale must be finite; got "${String(outputScale)}".`);
  }

  const paddedHiddenSize = padToQ4KBlock(inferredHiddenSize);
  const outputSize = batchSize * paddedHiddenSize * 4;
  const ownsResidualOutput = residualOutputBuffer == null;
  const ownsNormOutput = normOutputBuffer == null;
  const residualOutput = residualOutputBuffer ?? acquireBuffer(outputSize, undefined, 'residual_rmsnorm_pair_residual_output');
  const normOutput = normOutputBuffer ?? acquireBuffer(outputSize, undefined, 'residual_rmsnorm_pair_norm_output');
  const dispatchPlan = planRMSNormDispatch(target, batchSize);
  const variant = resolveResidualVariant();
  let completed = false;

  try {
    await unifiedKernelWrapper(
      'residual_rmsnorm_pair',
      target,
      variant,
      [input, residual, normWeightBuffer, residualOutput, normOutput],
      {
        hidden_size: inferredHiddenSize,
        num_tokens: batchSize,
        eps,
        output_scale: residualScale,
        token_stride: dispatchPlan.tokenStride,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
      },
      dispatchPlan.workgroups,
      {
        RMS_NORM_OFFSET: rmsNormWeightOffset,
        NORM_WEIGHT_IS_F16: normWeightDtype === 'f16',
      },
      null,
      resolveResidualDispatchLabel(label)
    );

    completed = true;
    return {
      residual: createTensor(residualOutput, 'f32', [batchSize, inferredHiddenSize], 'residual_rmsnorm_pair_residual_output'),
      nextInput: createTensor(normOutput, 'f32', [batchSize, inferredHiddenSize], 'residual_rmsnorm_pair_norm_output'),
    };
  } finally {
    if (!completed) {
      if (ownsResidualOutput) releaseBuffer(residualOutput);
      if (ownsNormOutput) releaseBuffer(normOutput);
    }
  }
}

export async function runResidualNextRMSNormPair(
  input,
  residual,
  normWeight,
  eps,
  options = {}
) {
  return runResidualRMSNormPairImpl(null, input, residual, normWeight, eps, options);
}

export async function recordResidualNextRMSNormPair(
  recorder,
  input,
  residual,
  normWeight,
  eps,
  options = {}
) {
  return runResidualRMSNormPairImpl(recorder, input, residual, normWeight, eps, options);
}
