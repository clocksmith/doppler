import { log } from '../../../debug/index.js';
import { createTensor } from '../../../gpu/tensor.js';
import {
  runAttention,
  runBiasAdd,
  runGeLU,
  runLayerNorm,
  runMatmul,
  runResidualAdd,
  runRMSNorm,
  runSiLU,
  runSplitQKV,
  runVisionRope2D,
  runVisionSpatialMerge,
} from '../../../gpu/kernel-selector.js';
import { acquireBuffer, releaseBuffer, uploadData } from '../../../memory/buffer-pool.js';
import { getBuffer, getWeightDtype, requireWeightDtype } from '../../../gpu/weight-buffer.js';
import { createImmediateResourceScope } from '../../resource-scope.js';
import { preprocessGlmOcrImage } from './glmocr-preprocess.js';

export {
  preprocessGlmOcrImage,
  resizeGlmOcrImageBicubic,
  resolveGlmOcrImageSize,
} from './glmocr-preprocess.js';

function requirePositiveInteger(value, label) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`[Vision] GLM-OCR requires ${label} to be a positive integer, got ${value}.`);
  }
  return value;
}

function createScope() {
  return createImmediateResourceScope({ release: releaseBuffer });
}

function registerTensor(scope, tensor, label) {
  scope.register(tensor.buffer, label, 'scopeOwned');
  return tensor;
}

function reshapeTensor(tensor, shape, label) {
  return createTensor(tensor.buffer, tensor.dtype, shape, label);
}

async function runLinear(scope, input, weight, bias, rows, inputDim, outputDim, label) {
  const output = registerTensor(scope, await runMatmul(
    input,
    weight,
    rows,
    outputDim,
    inputDim,
    { outputDtype: 'f32', transposeB: 'auto', role: label }
  ), label);
  if (bias) {
    await runBiasAdd(output, bias, rows, outputDim);
  }
  return reshapeTensor(output, [rows, outputDim], label);
}

async function runGlmOcrAttention(scope, hidden, layer, config, geometry, layerIndex) {
  const { numPatches, gridHeight, gridWidth } = geometry;
  const hiddenSize = config.hiddenSize;
  const numHeads = config.numHeads;
  const headDim = config.headDim;
  const qkv = await runLinear(
    scope,
    hidden,
    layer.qkvWeight,
    layer.qkvBias,
    numPatches,
    hiddenSize,
    3 * hiddenSize,
    `glmocr_vision_qkv_${layerIndex}`
  );
  const split = await runSplitQKV(qkv, {
    numTokens: numPatches,
    qSize: hiddenSize,
    kSize: hiddenSize,
    vSize: hiddenSize,
  });
  const q = registerTensor(scope, split.Q, `layer ${layerIndex} q`);
  const k = registerTensor(scope, split.K, `layer ${layerIndex} k`);
  const v = registerTensor(scope, split.V, `layer ${layerIndex} v`);
  scope.release(qkv.buffer);

  const qNorm = registerTensor(scope, await runRMSNorm(
    reshapeTensor(q, [numPatches * numHeads, headDim], `glmocr_vision_q_flat_${layerIndex}`),
    layer.qNormWeight,
    config.eps,
    { batchSize: numPatches * numHeads, hiddenSize: headDim }
  ), `layer ${layerIndex} normalized q`);
  const kNorm = registerTensor(scope, await runRMSNorm(
    reshapeTensor(k, [numPatches * numHeads, headDim], `glmocr_vision_k_flat_${layerIndex}`),
    layer.kNormWeight,
    config.eps,
    { batchSize: numPatches * numHeads, hiddenSize: headDim }
  ), `layer ${layerIndex} normalized k`);
  scope.release(q.buffer);
  scope.release(k.buffer);

  await runVisionRope2D(qNorm, {
    numTokens: numPatches,
    numHeads,
    headDim,
    gridHeight,
    gridWidth,
    ropeTheta: config.ropeTheta,
    spatialMergeSize: config.spatialMergeSize,
  });
  await runVisionRope2D(kNorm, {
    numTokens: numPatches,
    numHeads,
    headDim,
    gridHeight,
    gridWidth,
    ropeTheta: config.ropeTheta,
    spatialMergeSize: config.spatialMergeSize,
  });

  const attention = registerTensor(scope, await runAttention(
    reshapeTensor(qNorm, [numPatches, numHeads, headDim], `glmocr_vision_q_${layerIndex}`),
    reshapeTensor(kNorm, [numPatches, numHeads, headDim], `glmocr_vision_k_${layerIndex}`),
    reshapeTensor(v, [numPatches, numHeads, headDim], `glmocr_vision_v_${layerIndex}`),
    null,
    numHeads,
    headDim,
    {
      seqLen: numPatches,
      kvLen: numPatches,
      numKVHeads: numHeads,
      scale: 1 / Math.sqrt(headDim),
      causal: false,
      bidirectionalSpanStart: 0,
      bidirectionalSpanLength: 0,
      startPos: 0,
      outputBuffer: null,
      attnSoftcap: 0,
      slidingWindow: 0,
      kvLenBuffer: null,
      indirectBuffer: null,
      indirectOffset: 0,
      kvStart: 0,
      kvLayout: 'contiguous',
      kvPageTable: null,
      kvPageSize: 0,
      kernelPath: null,
      outputGate: null,
      useFlashPrefill: false,
      useOrtFlashPrefill: false,
    }
  ), `layer ${layerIndex} attention`);
  scope.release(qNorm.buffer);
  scope.release(kNorm.buffer);
  scope.release(v.buffer);

  const output = await runLinear(
    scope,
    reshapeTensor(attention, [numPatches, hiddenSize], `glmocr_vision_attention_flat_${layerIndex}`),
    layer.projWeight,
    layer.projBias,
    numPatches,
    hiddenSize,
    hiddenSize,
    `glmocr_vision_attention_output_${layerIndex}`
  );
  scope.release(attention.buffer);
  return output;
}

async function runGlmOcrMlp(scope, hidden, layer, config, numPatches, layerIndex) {
  const gate = await runLinear(
    scope,
    hidden,
    layer.gateProjWeight,
    layer.gateProjBias,
    numPatches,
    config.hiddenSize,
    config.intermediateSize,
    `glmocr_vision_gate_${layerIndex}`
  );
  const up = await runLinear(
    scope,
    hidden,
    layer.upProjWeight,
    layer.upProjBias,
    numPatches,
    config.hiddenSize,
    config.intermediateSize,
    `glmocr_vision_up_${layerIndex}`
  );
  const activated = registerTensor(scope, await runSiLU(up, {
    size: numPatches * config.intermediateSize,
    gate,
    swigluLimit: null,
    inputActivation: 'identity',
  }), `layer ${layerIndex} activated MLP`);
  scope.release(gate.buffer);
  scope.release(up.buffer);
  const output = await runLinear(
    scope,
    reshapeTensor(activated, [numPatches, config.intermediateSize], `glmocr_vision_mlp_${layerIndex}`),
    layer.downProjWeight,
    layer.downProjBias,
    numPatches,
    config.intermediateSize,
    config.hiddenSize,
    `glmocr_vision_down_${layerIndex}`
  );
  scope.release(activated.buffer);
  return output;
}

async function runGlmOcrMerger(scope, hidden, weights, config, outputTokens, probeTensor = null) {
  const projected = await runLinear(
    scope,
    hidden,
    weights.projWeight,
    null,
    outputTokens,
    config.outHiddenSize,
    config.outHiddenSize,
    'glmocr_vision_merger_projection'
  );
  await probeTensor?.('glmocr_vision_merger_projection', projected.buffer, {
    numTokens: outputTokens,
    hiddenSize: config.outHiddenSize,
    dtype: 'f32',
  });
  const normalized = registerTensor(scope, await runLayerNorm(
    projected,
    weights.postProjectionNormWeight,
    weights.postProjectionNormBias,
    config.eps,
    {
      batchSize: outputTokens,
      hiddenSize: config.outHiddenSize,
      normWeightDtype: requireWeightDtype(
        weights.postProjectionNormWeight,
        'GLM-OCR merger post-projection LayerNorm weight'
      ),
    }
  ), 'GLM-OCR merger normalized projection');
  await probeTensor?.('glmocr_vision_merger_normalized', normalized.buffer, {
    numTokens: outputTokens,
    hiddenSize: config.outHiddenSize,
    dtype: 'f32',
  });
  scope.release(projected.buffer);
  const gelu = registerTensor(scope, await runGeLU(normalized, {
    size: outputTokens * config.outHiddenSize,
  }), 'GLM-OCR merger GELU');
  await probeTensor?.('glmocr_vision_merger_gelu', gelu.buffer, {
    numTokens: outputTokens,
    hiddenSize: config.outHiddenSize,
    dtype: 'f32',
  });
  scope.release(normalized.buffer);

  const gate = await runLinear(
    scope,
    reshapeTensor(gelu, [outputTokens, config.outHiddenSize], 'glmocr_vision_merger_gelu'),
    weights.gateProjWeight,
    null,
    outputTokens,
    config.outHiddenSize,
    config.mergerIntermediateSize,
    'glmocr_vision_merger_gate'
  );
  await probeTensor?.('glmocr_vision_merger_gate', gate.buffer, {
    numTokens: outputTokens,
    hiddenSize: config.mergerIntermediateSize,
    dtype: 'f32',
  });
  const up = await runLinear(
    scope,
    reshapeTensor(gelu, [outputTokens, config.outHiddenSize], 'glmocr_vision_merger_gelu'),
    weights.upProjWeight,
    null,
    outputTokens,
    config.outHiddenSize,
    config.mergerIntermediateSize,
    'glmocr_vision_merger_up'
  );
  await probeTensor?.('glmocr_vision_merger_up', up.buffer, {
    numTokens: outputTokens,
    hiddenSize: config.mergerIntermediateSize,
    dtype: 'f32',
  });
  scope.release(gelu.buffer);
  const activated = registerTensor(scope, await runSiLU(up, {
    size: outputTokens * config.mergerIntermediateSize,
    gate,
    swigluLimit: null,
    inputActivation: 'identity',
  }), 'GLM-OCR merger gated activation');
  await probeTensor?.('glmocr_vision_merger_activated', activated.buffer, {
    numTokens: outputTokens,
    hiddenSize: config.mergerIntermediateSize,
    dtype: 'f32',
  });
  scope.release(gate.buffer);
  scope.release(up.buffer);
  const output = await runLinear(
    scope,
    reshapeTensor(activated, [outputTokens, config.mergerIntermediateSize], 'glmocr_vision_merger_activated'),
    weights.downProjWeight,
    null,
    outputTokens,
    config.mergerIntermediateSize,
    config.outHiddenSize,
    'glmocr_vision_merger_output'
  );
  await probeTensor?.('glmocr_vision_merger_output', output.buffer, {
    numTokens: outputTokens,
    hiddenSize: config.outHiddenSize,
    dtype: 'f32',
  });
  scope.release(activated.buffer);
  return output;
}

export async function encodeGlmOcrImage(params) {
  const { pixels, width, height, visionConfig, weights, probeTensor = null } = params;
  if (visionConfig.hiddenActivation !== 'silu') {
    throw new Error(
      `[Vision] GLM-OCR requires hiddenActivation="silu", got ${JSON.stringify(visionConfig.hiddenActivation)}.`
    );
  }
  const hiddenSize = requirePositiveInteger(Number(visionConfig.hiddenSize), 'hiddenSize');
  const outHiddenSize = requirePositiveInteger(Number(visionConfig.outHiddenSize), 'outHiddenSize');
  const downsampleKernelSize = requirePositiveInteger(
    Number(visionConfig.downsampleKernelSize),
    'downsampleKernelSize'
  );
  if (hiddenSize !== visionConfig.numHeads * visionConfig.headDim) {
    throw new Error(
      `[Vision] GLM-OCR attention geometry mismatch: hiddenSize=${hiddenSize}, ` +
      `numHeads=${visionConfig.numHeads}, headDim=${visionConfig.headDim}.`
    );
  }
  if (downsampleKernelSize !== visionConfig.spatialMergeSize) {
    throw new Error(
      `[Vision] GLM-OCR requires downsampleKernelSize to equal spatialMergeSize; ` +
      `got ${downsampleKernelSize}/${visionConfig.spatialMergeSize}.`
    );
  }
  if (outHiddenSize !== weights.textHiddenSize) {
    throw new Error(
      `[Vision] GLM-OCR output width ${outHiddenSize} does not match text hidden size ${weights.textHiddenSize}.`
    );
  }
  if (!Array.isArray(weights.layers) || weights.layers.length !== visionConfig.depth) {
    throw new Error(
      `[Vision] GLM-OCR requires ${visionConfig.depth} loaded encoder layers, got ${weights.layers?.length ?? 'missing'}.`
    );
  }

  const preprocessed = preprocessGlmOcrImage(pixels, width, height, visionConfig);
  const outputTokens = preprocessed.numPatches / (visionConfig.spatialMergeSize ** 2);
  if (!Number.isInteger(outputTokens) || outputTokens <= 0) {
    throw new Error(`[Vision] GLM-OCR produced invalid output token count ${outputTokens}.`);
  }
  if (outputTokens > visionConfig.defaultOutputLength) {
    throw new Error(
      `[Vision] GLM-OCR produced ${outputTokens} output tokens, exceeding ` +
      `defaultOutputLength=${visionConfig.defaultOutputLength}.`
    );
  }
  log.debug(
    'Vision',
    `glmocr encode: ${width}x${height} -> ${preprocessed.gridWidth}x${preprocessed.gridHeight} ` +
    `patches=${preprocessed.numPatches} outputTokens=${outputTokens}`
  );

  const scope = createScope();
  let succeeded = false;
  try {
    await probeTensor?.('glmocr_vision_patch_input', preprocessed.patches, {
      numTokens: preprocessed.numPatches,
      hiddenSize: preprocessed.patchDim,
      dtype: 'f32',
    });
    const patchBuffer = acquireBuffer(
      preprocessed.patches.byteLength,
      undefined,
      'glmocr_vision_patches'
    );
    scope.register(patchBuffer, 'GLM-OCR patch data', 'scopeOwned');
    uploadData(patchBuffer, preprocessed.patches, 0);
    const patchTensor = createTensor(
      patchBuffer,
      'f32',
      [preprocessed.numPatches, preprocessed.patchDim],
      'glmocr_vision_patches'
    );
    await probeTensor?.('glmocr_vision_patch_bias', getBuffer(weights.patchProjBias), {
      numTokens: 1,
      hiddenSize,
      dtype: getWeightDtype(weights.patchProjBias),
    });
    let hidden = await runLinear(
      scope,
      patchTensor,
      weights.patchProjWeight,
      weights.patchProjBias,
      preprocessed.numPatches,
      preprocessed.patchDim,
      hiddenSize,
      'glmocr_vision_patch_projection'
    );
    scope.release(patchBuffer);
    await probeTensor?.('glmocr_vision_patch_projection', hidden.buffer, {
      numTokens: preprocessed.numPatches,
      hiddenSize,
      dtype: 'f32',
    });

    for (let layerIndex = 0; layerIndex < weights.layers.length; layerIndex++) {
      const layer = weights.layers[layerIndex];
      const norm1 = registerTensor(scope, await runRMSNorm(
        hidden,
        layer.norm1Weight,
        visionConfig.eps,
        { batchSize: preprocessed.numPatches, hiddenSize }
      ), `layer ${layerIndex} norm1`);
      const attention = await runGlmOcrAttention(
        scope,
        norm1,
        layer,
        visionConfig,
        preprocessed,
        layerIndex
      );
      scope.release(norm1.buffer);
      const attentionResidual = registerTensor(scope, await runResidualAdd(
        hidden,
        attention,
        preprocessed.numPatches * hiddenSize
      ), `layer ${layerIndex} attention residual`);
      scope.release(hidden.buffer);
      scope.release(attention.buffer);
      hidden = reshapeTensor(
        attentionResidual,
        [preprocessed.numPatches, hiddenSize],
        `glmocr_vision_hidden_attention_${layerIndex}`
      );
      await probeTensor?.('glmocr_vision_attention_residual', hidden.buffer, {
        layerIdx: layerIndex,
        numTokens: preprocessed.numPatches,
        hiddenSize,
        dtype: 'f32',
      });

      const norm2 = registerTensor(scope, await runRMSNorm(
        hidden,
        layer.norm2Weight,
        visionConfig.eps,
        { batchSize: preprocessed.numPatches, hiddenSize }
      ), `layer ${layerIndex} norm2`);
      const mlp = await runGlmOcrMlp(
        scope,
        norm2,
        layer,
        visionConfig,
        preprocessed.numPatches,
        layerIndex
      );
      scope.release(norm2.buffer);
      const mlpResidual = registerTensor(scope, await runResidualAdd(
        hidden,
        mlp,
        preprocessed.numPatches * hiddenSize
      ), `layer ${layerIndex} MLP residual`);
      scope.release(hidden.buffer);
      scope.release(mlp.buffer);
      hidden = reshapeTensor(
        mlpResidual,
        [preprocessed.numPatches, hiddenSize],
        `glmocr_vision_hidden_mlp_${layerIndex}`
      );
      await probeTensor?.('glmocr_vision_layer_out', hidden.buffer, {
        layerIdx: layerIndex,
        numTokens: preprocessed.numPatches,
        hiddenSize,
        dtype: 'f32',
      });
    }

    const postNorm = registerTensor(scope, await runRMSNorm(
      hidden,
      weights.postLayerNorm,
      visionConfig.eps,
      { batchSize: preprocessed.numPatches, hiddenSize }
    ), 'GLM-OCR post encoder norm');
    scope.release(hidden.buffer);
    hidden = null;
    await probeTensor?.('glmocr_vision_postnorm', postNorm.buffer, {
      numTokens: preprocessed.numPatches,
      hiddenSize,
      dtype: 'f32',
    });

    const gathered = registerTensor(scope, await runVisionSpatialMerge(postNorm, {
      gridHeight: preprocessed.gridHeight,
      gridWidth: preprocessed.gridWidth,
      hiddenSize,
      mergeSize: visionConfig.spatialMergeSize,
      channelFirst: true,
      inputBlockMajor: true,
    }), 'GLM-OCR channel-first downsample input');
    scope.release(postNorm.buffer);
    await probeTensor?.('glmocr_vision_spatial_merge', gathered.buffer, {
      numTokens: outputTokens,
      hiddenSize: hiddenSize * downsampleKernelSize * downsampleKernelSize,
      dtype: 'f32',
    });
    const downsampled = await runLinear(
      scope,
      gathered,
      weights.downsampleWeight,
      weights.downsampleBias,
      outputTokens,
      hiddenSize * downsampleKernelSize * downsampleKernelSize,
      outHiddenSize,
      'glmocr_vision_downsample'
    );
    scope.release(gathered.buffer);
    await probeTensor?.('glmocr_vision_downsample', downsampled.buffer, {
      numTokens: outputTokens,
      hiddenSize: outHiddenSize,
      dtype: 'f32',
    });

    const merged = await runGlmOcrMerger(
      scope,
      downsampled,
      weights.merger,
      visionConfig,
      outputTokens,
      probeTensor
    );
    scope.release(downsampled.buffer);
    scope.retain(merged.buffer, 'GLM-OCR output', 'caller owns encoded image features');
    succeeded = true;
    return {
      features: merged.buffer,
      numTokens: outputTokens,
      gridThw: [1, preprocessed.gridHeight, preprocessed.gridWidth],
      imageWidth: preprocessed.targetWidth,
      imageHeight: preprocessed.targetHeight,
    };
  } finally {
    scope.close(succeeded ? 'success' : 'failure');
  }
}
