import { log } from '../../../debug/index.js';
import { createTensor } from '../../../gpu/tensor.js';
import { runVisionAveragePool } from '../../../gpu/kernels/vision-average-pool.js';
import { runVisionPositionEmbedding } from '../../../gpu/kernels/vision-position-embedding.js';
import { runVisionRope2D } from '../../../gpu/kernels/vision-rope-2d.js';
import {
  runAttention,
  runGeLU,
  runMatmul,
  runResidualAdd,
  runRMSNorm,
} from '../../../gpu/kernel-selector.js';
import { acquireBuffer, releaseBuffer, uploadData } from '../../../memory/buffer-pool.js';
import { createImmediateResourceScope } from '../../resource-scope.js';
import { getQKNormOnesBuffer } from '../text/attention/types.js';
import { shouldClamp, runClippableLinear } from '../shared/clipped-linear.js';

function createTensorFromBuffer(buffer, shape, label) {
  return createTensor(buffer, 'f32', shape, label);
}

function reshapeTensor(tensor, shape, label) {
  return createTensor(tensor.buffer, tensor.dtype, shape, label);
}

function createVisionResourceScope() {
  return createImmediateResourceScope({ release: releaseBuffer });
}

function registerTensor(scope, tensor, label) {
  scope.register(tensor.buffer, label, 'scopeOwned');
  return tensor;
}

function requirePositiveInteger(value, label) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`[Vision] Gemma 4 requires ${label} to be a positive integer, got ${value}.`);
  }
  return value;
}

function resolveSourceChannels(pixels, width, height) {
  const area = width * height;
  if (!Number.isFinite(area) || area <= 0) {
    throw new Error(`[Vision] Invalid image size ${width}x${height}.`);
  }
  const channels = pixels.length / area;
  if (!Number.isFinite(channels) || Math.floor(channels) !== channels || (channels !== 3 && channels !== 4)) {
    throw new Error(
      `[Vision] Expected interleaved RGB or RGBA pixels, got length=${pixels.length} for ${width}x${height}.`
    );
  }
  return channels;
}

function getPixelValue(pixels, srcChannels, index) {
  const value = pixels[index];
  if (pixels instanceof Float32Array) {
    return value <= 1.0 ? value : (value / 255.0);
  }
  if (pixels instanceof Uint8Array || pixels instanceof Uint8ClampedArray) {
    return value / 255.0;
  }
  return Number(value) / 255.0;
}

function clamp01(value) {
  if (value <= 0) return 0;
  if (value >= 1) return 1;
  return value;
}

function cubicWeight(distance) {
  const a = -0.5;
  const x = Math.abs(distance);
  if (x <= 1) {
    return ((a + 2) * x * x * x) - ((a + 3) * x * x) + 1;
  }
  if (x < 2) {
    return (a * x * x * x) - (5 * a * x * x) + (8 * a * x) - (4 * a);
  }
  return 0;
}

function resizeImageToRgbFloat32(pixels, width, height, targetWidth, targetHeight) {
  const srcChannels = resolveSourceChannels(pixels, width, height);
  const out = new Float32Array(targetWidth * targetHeight * 3);
  const scaleX = width / targetWidth;
  const scaleY = height / targetHeight;

  for (let y = 0; y < targetHeight; y++) {
    const srcY = ((y + 0.5) * scaleY) - 0.5;
    const yBase = Math.floor(srcY);

    for (let x = 0; x < targetWidth; x++) {
      const srcX = ((x + 0.5) * scaleX) - 0.5;
      const xBase = Math.floor(srcX);

      for (let c = 0; c < 3; c++) {
        let weightedValue = 0;
        let weightSum = 0;

        for (let sampleY = -1; sampleY <= 2; sampleY++) {
          const srcSampleY = yBase + sampleY;
          const clampedY = Math.max(0, Math.min(srcSampleY, height - 1));
          const yWeight = cubicWeight(srcY - srcSampleY);
          if (yWeight === 0) continue;

          for (let sampleX = -1; sampleX <= 2; sampleX++) {
            const srcSampleX = xBase + sampleX;
            const clampedX = Math.max(0, Math.min(srcSampleX, width - 1));
            const xWeight = cubicWeight(srcX - srcSampleX);
            if (xWeight === 0) continue;

            const weight = xWeight * yWeight;
            const idx = (clampedY * width + clampedX) * srcChannels + c;
            weightedValue += getPixelValue(pixels, srcChannels, idx) * weight;
            weightSum += weight;
          }
        }

        const outputValue = weightSum === 0
          ? getPixelValue(
            pixels,
            srcChannels,
            (Math.max(0, Math.min(yBase, height - 1)) * width + Math.max(0, Math.min(xBase, width - 1))) * srcChannels + c
          )
          : (weightedValue / weightSum);
        out[(y * targetWidth + x) * 3 + c] = clamp01(outputValue);
      }
    }
  }

  return out;
}

function getAspectRatioPreservingSize(height, width, patchSize, maxPatches, poolingKernelSize) {
  const totalPixels = height * width;
  const targetPixels = maxPatches * (patchSize ** 2);
  const factor = Math.sqrt(targetPixels / totalPixels);
  const idealHeight = factor * height;
  const idealWidth = factor * width;
  const sideMultiple = poolingKernelSize * patchSize;

  let targetHeight = Math.floor(idealHeight / sideMultiple) * sideMultiple;
  let targetWidth = Math.floor(idealWidth / sideMultiple) * sideMultiple;

  if (targetHeight === 0 && targetWidth === 0) {
    throw new Error(
      `[Vision] Image resized to 0x0. Check patchSize=${patchSize} and poolingKernelSize=${poolingKernelSize}.`
    );
  }

  const maxSideLength = Math.floor(maxPatches / (poolingKernelSize ** 2)) * sideMultiple;
  if (targetHeight === 0) {
    targetHeight = sideMultiple;
    targetWidth = Math.min(Math.floor(width / height) * sideMultiple, maxSideLength);
  } else if (targetWidth === 0) {
    targetWidth = sideMultiple;
    targetHeight = Math.min(Math.floor(height / width) * sideMultiple, maxSideLength);
  }

  if (targetHeight * targetWidth > targetPixels) {
    throw new Error(
      `[Vision] Resizing ${width}x${height} -> ${targetWidth}x${targetHeight} exceeds max patch budget ${maxPatches}.`
    );
  }

  return { targetHeight, targetWidth };
}

export function preprocessGemma4Image(pixels, width, height, visionConfig, softTokenBudget) {
  const patchSize = Number(visionConfig.patchSize);
  const poolingKernelSize = Number(visionConfig.poolingKernelSize);
  if (!Number.isFinite(poolingKernelSize) || poolingKernelSize < 1) {
    throw new Error(
      `[Vision] Gemma 4 requires vision_config.pooling_kernel_size to be a positive integer, got ${visionConfig.poolingKernelSize}.`
    );
  }
  const effectiveBudget = softTokenBudget ?? visionConfig.defaultOutputLength;
  if (softTokenBudget != null) {
    const tiers = visionConfig.softTokenBudgetTiers;
    if (Array.isArray(tiers) && tiers.length > 0 && !tiers.includes(softTokenBudget)) {
      throw new Error(
        `[Vision] softTokenBudget=${softTokenBudget} is not in the allowed tiers [${tiers.join(', ')}].`
      );
    }
  }
  const maxSoftTokens = Number(effectiveBudget);
  if (!Number.isFinite(maxSoftTokens) || maxSoftTokens < 1 || Math.floor(maxSoftTokens) !== maxSoftTokens) {
    throw new Error(
      `[Vision] Gemma 4 requires a positive integer soft token budget, got ${effectiveBudget}.`
    );
  }
  const maxPatches = maxSoftTokens * (poolingKernelSize ** 2);
  const { targetHeight, targetWidth } = getAspectRatioPreservingSize(
    height,
    width,
    patchSize,
    maxPatches,
    poolingKernelSize
  );

  const resized = resizeImageToRgbFloat32(pixels, width, height, targetWidth, targetHeight);
  const gridHeight = targetHeight / patchSize;
  const gridWidth = targetWidth / patchSize;
  const numPatches = gridHeight * gridWidth;
  const patchArea = 3 * patchSize * patchSize;
  const patches = new Float32Array(numPatches * patchArea);
  const positions = new Int32Array(numPatches * 2);

  for (let patchY = 0; patchY < gridHeight; patchY++) {
    for (let patchX = 0; patchX < gridWidth; patchX++) {
      const patchIdx = patchY * gridWidth + patchX;
      positions[patchIdx * 2] = patchX;
      positions[patchIdx * 2 + 1] = patchY;

      let dstOffset = patchIdx * patchArea;
      for (let localY = 0; localY < patchSize; localY++) {
        for (let localX = 0; localX < patchSize; localX++) {
          const srcPixelOffset = ((patchY * patchSize + localY) * targetWidth + (patchX * patchSize + localX)) * 3;
          // Gemma 4 vision preprocessing rescales pixels to [0, 1] without extra normalization.
          patches[dstOffset++] = resized[srcPixelOffset];
          patches[dstOffset++] = resized[srcPixelOffset + 1];
          patches[dstOffset++] = resized[srcPixelOffset + 2];
        }
      }
    }
  }

  return {
    patches,
    positions,
    gridHeight,
    gridWidth,
    numPatches,
    outputLength: numPatches / (poolingKernelSize ** 2),
  };
}

async function runVisionAttention(hiddenTensor, layerWeights, visionConfig, geometry, numTokens, hiddenSize) {
  const numHeads = requirePositiveInteger(Number(visionConfig.numHeads), 'numHeads');
  const numKVHeads = requirePositiveInteger(Number(visionConfig.numKeyValueHeads), 'numKeyValueHeads');
  const headDim = requirePositiveInteger(Number(visionConfig.headDim), 'headDim');
  if (hiddenSize !== numHeads * headDim) {
    throw new Error(
      `[Vision] Gemma 4 attention geometry mismatch: hiddenSize=${hiddenSize}, ` +
      `numHeads=${numHeads}, headDim=${headDim}.`
    );
  }
  if (numHeads % numKVHeads !== 0) {
    throw new Error(
      `[Vision] Gemma 4 requires numHeads divisible by numKeyValueHeads, got ${numHeads}/${numKVHeads}.`
    );
  }

  const scope = createVisionResourceScope();
  let succeeded = false;
  try {
    const qTensor = registerTensor(scope, await runClippableLinear(
      hiddenTensor,
      layerWeights.qProj,
      numTokens,
      numHeads * headDim,
      hiddenSize,
      layerWeights.qProjClip,
      'gemma4_vision_q_proj'
    ), 'q projection');
    const kTensor = registerTensor(scope, await runClippableLinear(
      hiddenTensor,
      layerWeights.kProj,
      numTokens,
      numKVHeads * headDim,
      hiddenSize,
      layerWeights.kProjClip,
      'gemma4_vision_k_proj'
    ), 'k projection');
    const vTensor = registerTensor(scope, await runClippableLinear(
      hiddenTensor,
      layerWeights.vProj,
      numTokens,
      numKVHeads * headDim,
      hiddenSize,
      layerWeights.vProjClip,
      'gemma4_vision_v_proj'
    ), 'v projection');

    const qNormTensor = registerTensor(scope, await runRMSNorm(
      reshapeTensor(qTensor, [numTokens * numHeads, headDim], 'gemma4_vision_q_flat'),
      layerWeights.qNorm,
      visionConfig.eps,
      { batchSize: numTokens * numHeads, hiddenSize: headDim }
    ), 'normalized q');
    const kNormTensor = registerTensor(scope, await runRMSNorm(
      reshapeTensor(kTensor, [numTokens * numKVHeads, headDim], 'gemma4_vision_k_flat'),
      layerWeights.kNorm,
      visionConfig.eps,
      { batchSize: numTokens * numKVHeads, hiddenSize: headDim }
    ), 'normalized k');
    const vNormTensor = registerTensor(scope, await runRMSNorm(
      reshapeTensor(vTensor, [numTokens * numKVHeads, headDim], 'gemma4_vision_v_flat'),
      getQKNormOnesBuffer(headDim),
      visionConfig.eps,
      { batchSize: numTokens * numKVHeads, hiddenSize: headDim }
    ), 'normalized v');

    scope.release(qTensor.buffer);
    scope.release(kTensor.buffer);
    scope.release(vTensor.buffer);

    await runVisionRope2D(qNormTensor, {
      numTokens,
      numHeads,
      headDim,
      gridHeight: geometry.gridHeight,
      gridWidth: geometry.gridWidth,
      ropeTheta: geometry.ropeTheta,
    });
    await runVisionRope2D(kNormTensor, {
      numTokens,
      numHeads: numKVHeads,
      headDim,
      gridHeight: geometry.gridHeight,
      gridWidth: geometry.gridWidth,
      ropeTheta: geometry.ropeTheta,
    });

    const attnTensor = await runAttention(
      reshapeTensor(qNormTensor, [numTokens, numHeads, headDim], 'gemma4_vision_q'),
      reshapeTensor(kNormTensor, [numTokens, numKVHeads, headDim], 'gemma4_vision_k'),
      reshapeTensor(vNormTensor, [numTokens, numKVHeads, headDim], 'gemma4_vision_v'),
      null,
      numHeads,
      headDim,
      {
        seqLen: numTokens,
        kvLen: numTokens,
        numKVHeads,
        scale: 1.0,
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
    );
    registerTensor(scope, attnTensor, 'attention output');

    scope.release(qNormTensor.buffer);
    scope.release(kNormTensor.buffer);
    scope.release(vNormTensor.buffer);

    const output = registerTensor(scope, await runClippableLinear(
      reshapeTensor(attnTensor, [numTokens, hiddenSize], 'gemma4_vision_attn_flat'),
      layerWeights.oProj,
      numTokens,
      hiddenSize,
      hiddenSize,
      layerWeights.oProjClip,
      'gemma4_vision_o_proj'
    ), 'attention projection');
    scope.release(attnTensor.buffer);
    scope.retain(output.buffer, 'attention projection', 'caller owns attention output');
    succeeded = true;
    return output;
  } finally {
    scope.close(succeeded ? 'success' : 'failure');
  }
}

async function runVisionMlp(hiddenTensor, layerWeights, visionConfig, numTokens, hiddenSize) {
  const intermediateSize = requirePositiveInteger(Number(visionConfig.intermediateSize), 'intermediateSize');
  const scope = createVisionResourceScope();
  let succeeded = false;
  try {
    const gateTensor = registerTensor(scope, await runClippableLinear(
      hiddenTensor,
      layerWeights.gateProj,
      numTokens,
      intermediateSize,
      hiddenSize,
      layerWeights.gateProjClip,
      'gemma4_vision_gate_proj'
    ), 'MLP gate projection');
    const upTensor = registerTensor(scope, await runClippableLinear(
      hiddenTensor,
      layerWeights.upProj,
      numTokens,
      intermediateSize,
      hiddenSize,
      layerWeights.upProjClip,
      'gemma4_vision_up_proj'
    ), 'MLP up projection');
    const activatedTensor = registerTensor(scope, await runGeLU(gateTensor, {
      size: numTokens * intermediateSize,
      gate: upTensor,
    }), 'MLP activation');
    scope.release(gateTensor.buffer);
    scope.release(upTensor.buffer);

    const output = registerTensor(scope, await runClippableLinear(
      activatedTensor,
      layerWeights.downProj,
      numTokens,
      hiddenSize,
      intermediateSize,
      layerWeights.downProjClip,
      'gemma4_vision_down_proj'
    ), 'MLP down projection');
    scope.release(activatedTensor.buffer);
    scope.retain(output.buffer, 'MLP down projection', 'caller owns MLP output');
    succeeded = true;
    return output;
  } finally {
    scope.close(succeeded ? 'success' : 'failure');
  }
}

export async function encodeGemma4Image(params) {
  const { pixels, width, height, visionConfig, weights, softTokenBudget } = params;
  const hiddenActivation = String(visionConfig.hiddenActivation ?? '').trim();
  if (hiddenActivation !== 'gelu' && hiddenActivation !== 'gelu_pytorch_tanh') {
    throw new Error(
      `[Vision] Gemma 4 vision hiddenActivation must be "gelu" or "gelu_pytorch_tanh", got ${JSON.stringify(visionConfig.hiddenActivation)}.`
    );
  }
  if (visionConfig.standardize === true) {
    throw new Error('[Vision] Gemma 4 standardize=true is not supported by the current runtime.');
  }
  if (visionConfig.useClippedLinears !== true) {
    throw new Error('[Vision] Gemma 4 vision runtime requires useClippedLinears=true.');
  }
  const hiddenSize = requirePositiveInteger(Number(visionConfig.hiddenSize), 'hiddenSize');
  const patchSize = requirePositiveInteger(Number(visionConfig.patchSize), 'patchSize');
  const poolingKernelSize = requirePositiveInteger(Number(visionConfig.poolingKernelSize), 'poolingKernelSize');
  const ropeTheta = Number(visionConfig.ropeTheta);
  if (!Number.isFinite(ropeTheta) || ropeTheta <= 0) {
    throw new Error(
      `[Vision] Gemma 4 requires a positive ropeTheta, got ${JSON.stringify(visionConfig.ropeTheta)}.`
    );
  }

  const preprocessed = preprocessGemma4Image(pixels, width, height, visionConfig, softTokenBudget);
  log.debug(
    'Vision',
    `gemma4 encode: ${width}x${height} -> ${preprocessed.gridWidth}x${preprocessed.gridHeight} patches=${preprocessed.numPatches}`
  );
  const scaledPatches = new Float32Array(preprocessed.patches.length);
  for (let index = 0; index < preprocessed.patches.length; index++) {
    scaledPatches[index] = 2.0 * (preprocessed.patches[index] - 0.5);
  }

  const scope = createVisionResourceScope();
  let succeeded = false;
  try {
    const patchTensor = registerTensor(scope, createTensorFromBuffer(
      acquireBuffer(scaledPatches.byteLength, undefined, 'gemma4_vision_patches'),
      [preprocessed.numPatches, 3 * patchSize * patchSize],
      'gemma4_vision_patches'
    ), 'patch tensor');
    uploadData(patchTensor.buffer, scaledPatches, 0);

    const positionTensor = registerTensor(scope, await runVisionPositionEmbedding(
      weights.patchPositionEmbeddingTable,
      {
        gridHeight: preprocessed.gridHeight,
        gridWidth: preprocessed.gridWidth,
        positionEmbeddingSize: Number(visionConfig.positionEmbeddingSize),
        hiddenSize,
      }
    ), 'position embedding');

    let hiddenTensor = registerTensor(scope, await runMatmul(
      patchTensor,
      weights.patchInputProj,
      preprocessed.numPatches,
      hiddenSize,
      3 * patchSize * patchSize,
      { outputDtype: 'f32', transposeB: 'auto' }
    ), 'patch projection');
    scope.release(patchTensor.buffer);

    const embedded = registerTensor(
      scope,
      await runResidualAdd(hiddenTensor, positionTensor, preprocessed.numPatches * hiddenSize),
      'positioned patches'
    );
    scope.release(hiddenTensor.buffer);
    scope.release(positionTensor.buffer);
    hiddenTensor = reshapeTensor(embedded, [preprocessed.numPatches, hiddenSize], 'gemma4_vision_hidden_0');

    for (let layerIdx = 0; layerIdx < weights.layers.length; layerIdx++) {
      const layerWeights = weights.layers[layerIdx];
      const inputNorm = registerTensor(scope, await runRMSNorm(
        hiddenTensor,
        layerWeights.inputLayerNorm,
        visionConfig.eps,
        { batchSize: preprocessed.numPatches, hiddenSize }
      ), `layer ${layerIdx} input norm`);

      const attnOut = registerTensor(scope, await runVisionAttention(
        inputNorm,
        layerWeights,
        visionConfig,
        {
          gridHeight: preprocessed.gridHeight,
          gridWidth: preprocessed.gridWidth,
          ropeTheta,
        },
        preprocessed.numPatches,
        hiddenSize
      ), `layer ${layerIdx} attention`);
      scope.release(inputNorm.buffer);

      const postAttnNorm = registerTensor(scope, await runRMSNorm(
        attnOut,
        layerWeights.postAttentionLayerNorm,
        visionConfig.eps,
        { batchSize: preprocessed.numPatches, hiddenSize }
      ), `layer ${layerIdx} post-attention norm`);
      scope.release(attnOut.buffer);

      const attnResidual = registerTensor(scope, await runResidualAdd(
        hiddenTensor,
        postAttnNorm,
        preprocessed.numPatches * hiddenSize
      ), `layer ${layerIdx} attention residual`);
      scope.release(hiddenTensor.buffer);
      scope.release(postAttnNorm.buffer);
      hiddenTensor = reshapeTensor(attnResidual, [preprocessed.numPatches, hiddenSize], `gemma4_vision_hidden_attn_${layerIdx}`);

      const preFfNorm = registerTensor(scope, await runRMSNorm(
        hiddenTensor,
        layerWeights.preFeedforwardLayerNorm,
        visionConfig.eps,
        { batchSize: preprocessed.numPatches, hiddenSize }
      ), `layer ${layerIdx} pre-FFN norm`);
      const mlpOut = registerTensor(
        scope,
        await runVisionMlp(preFfNorm, layerWeights, visionConfig, preprocessed.numPatches, hiddenSize),
        `layer ${layerIdx} MLP`
      );
      scope.release(preFfNorm.buffer);

      const postFfNorm = registerTensor(scope, await runRMSNorm(
        mlpOut,
        layerWeights.postFeedforwardLayerNorm,
        visionConfig.eps,
        { batchSize: preprocessed.numPatches, hiddenSize }
      ), `layer ${layerIdx} post-FFN norm`);
      scope.release(mlpOut.buffer);

      const ffResidual = registerTensor(scope, await runResidualAdd(
        hiddenTensor,
        postFfNorm,
        preprocessed.numPatches * hiddenSize
      ), `layer ${layerIdx} FFN residual`);
      scope.release(hiddenTensor.buffer);
      scope.release(postFfNorm.buffer);
      hiddenTensor = reshapeTensor(ffResidual, [preprocessed.numPatches, hiddenSize], `gemma4_vision_hidden_ff_${layerIdx}`);
    }

    const pooled = registerTensor(scope, await runVisionAveragePool(hiddenTensor, {
      gridHeight: preprocessed.gridHeight,
      gridWidth: preprocessed.gridWidth,
      hiddenSize,
      poolingSize: poolingKernelSize,
    }), 'pooled vision state');
    scope.release(hiddenTensor.buffer);
    hiddenTensor = null;
    const outputLength = preprocessed.outputLength;
    const pooledTensor = registerTensor(scope, await runRMSNorm(
      pooled,
      getQKNormOnesBuffer(hiddenSize),
      visionConfig.eps,
      { batchSize: outputLength, hiddenSize }
    ), 'normalized pooled vision state');
    scope.release(pooled.buffer);

    let projected = null;
    if (weights.projector) {
      projected = registerTensor(scope, await runMatmul(
        pooledTensor,
        weights.projector,
        outputLength,
        weights.textHiddenSize,
        hiddenSize,
        { outputDtype: 'f32', transposeB: 'auto' }
      ), 'vision projector output');
      scope.release(pooledTensor.buffer);
    } else {
      if (hiddenSize !== weights.textHiddenSize) {
        throw new Error(
          `[Vision] Gemma 4 vision encoder-free mode has no projector, but vision hiddenSize (${hiddenSize}) ` +
          `does not match text hiddenSize (${weights.textHiddenSize}).`
        );
      }
      projected = pooledTensor;
    }

    scope.retain(projected.buffer, 'vision output', 'caller owns encoded vision features');
    succeeded = true;
    return {
      features: projected.buffer,
      numTokens: outputLength,
      gridThw: [1, preprocessed.gridHeight, preprocessed.gridWidth],
      imageWidth: preprocessed.gridWidth * patchSize,
      imageHeight: preprocessed.gridHeight * patchSize,
    };
  } finally {
    scope.close(succeeded ? 'success' : 'failure');
  }
}
