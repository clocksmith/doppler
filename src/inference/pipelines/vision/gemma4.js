import { log } from '../../../debug/index.js';
import { getDevice } from '../../../gpu/device.js';
import { createTensor } from '../../../gpu/tensor.js';
import {
  runAttention,
  runGeLU,
  runMatmul,
  runResidualAdd,
  runRMSNorm,
} from '../../../gpu/kernel-selector.js';
import { acquireBuffer, readBuffer, releaseBuffer, uploadData } from '../../../memory/buffer-pool.js';
import { getQKNormOnesBuffer } from '../text/attention/types.js';
import { shouldClamp, runClippableLinear } from '../shared/clipped-linear.js';

function createTensorFromBuffer(buffer, shape, label) {
  return createTensor(buffer, 'f32', shape, label);
}

function reshapeTensor(tensor, shape, label) {
  return createTensor(tensor.buffer, tensor.dtype, shape, label);
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

function buildPatchPositionEmbeddings(positionTable, positions, positionEmbeddingSize, hiddenSize) {
  const numPatches = positions.length / 2;
  const output = new Float32Array(numPatches * hiddenSize);
  const tableStride = positionEmbeddingSize * hiddenSize;

  for (let patchIdx = 0; patchIdx < numPatches; patchIdx++) {
    const x = positions[patchIdx * 2];
    const y = positions[patchIdx * 2 + 1];
    if (x < 0 || y < 0 || x >= positionEmbeddingSize || y >= positionEmbeddingSize) {
      throw new Error(
        `[Vision] Patch position (${x}, ${y}) exceeds position embedding size ${positionEmbeddingSize}.`
      );
    }
    const dstBase = patchIdx * hiddenSize;
    const xBase = x * hiddenSize;
    const yBase = tableStride + y * hiddenSize;
    for (let d = 0; d < hiddenSize; d++) {
      output[dstBase + d] = positionTable[xBase + d] + positionTable[yBase + d];
    }
  }

  return output;
}

async function readTensorF32(tensor, expectedLength, label) {
  const bytes = await readBuffer(tensor.buffer, expectedLength * Float32Array.BYTES_PER_ELEMENT);
  const copied = bytes instanceof ArrayBuffer
    ? bytes.slice(0)
    : bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const result = new Float32Array(copied);
  if (result.length !== expectedLength) {
    throw new Error(`[Vision] ${label}: expected ${expectedLength} floats, got ${result.length}.`);
  }
  return result;
}

function buildVisionRopeCache(positions, headDim, ropeTheta) {
  if (headDim % 4 !== 0) {
    throw new Error(`[Vision] headDim=${headDim} is incompatible with Gemma4 multidimensional RoPE.`);
  }
  const numTokens = positions.length / 2;
  const spatialDim = headDim / 2;
  const halfRotary = spatialDim / 2;
  const invFreq = new Float32Array(halfRotary);
  for (let i = 0; i < halfRotary; i++) {
    invFreq[i] = 1.0 / (ropeTheta ** ((2 * i) / spatialDim));
  }

  const cosX = new Float32Array(numTokens * halfRotary);
  const sinX = new Float32Array(numTokens * halfRotary);
  const cosY = new Float32Array(numTokens * halfRotary);
  const sinY = new Float32Array(numTokens * halfRotary);

  for (let tokenIdx = 0; tokenIdx < numTokens; tokenIdx++) {
    const x = positions[tokenIdx * 2];
    const y = positions[tokenIdx * 2 + 1];
    const base = tokenIdx * halfRotary;
    for (let i = 0; i < halfRotary; i++) {
      const angleX = x * invFreq[i];
      const angleY = y * invFreq[i];
      cosX[base + i] = Math.cos(angleX);
      sinX[base + i] = Math.sin(angleX);
      cosY[base + i] = Math.cos(angleY);
      sinY[base + i] = Math.sin(angleY);
    }
  }

  return {
    cosX,
    sinX,
    cosY,
    sinY,
    halfRotary,
  };
}

function applyVisionRopeInPlace(data, ropeCache, numTokens, numHeads, headDim) {
  const tokenStride = numHeads * headDim;
  const spatialOffset = headDim / 2;
  const pairCount = ropeCache.halfRotary;

  for (let tokenIdx = 0; tokenIdx < numTokens; tokenIdx++) {
    const ropeBase = tokenIdx * pairCount;
    for (let headIdx = 0; headIdx < numHeads; headIdx++) {
      const base = tokenIdx * tokenStride + headIdx * headDim;

      for (let i = 0; i < pairCount; i++) {
        const cos = ropeCache.cosX[ropeBase + i];
        const sin = ropeCache.sinX[ropeBase + i];
        const a = data[base + i];
        const b = data[base + pairCount + i];
        data[base + i] = (a * cos) - (b * sin);
        data[base + pairCount + i] = (b * cos) + (a * sin);
      }

      for (let i = 0; i < pairCount; i++) {
        const cos = ropeCache.cosY[ropeBase + i];
        const sin = ropeCache.sinY[ropeBase + i];
        const partBase = base + spatialOffset;
        const a = data[partBase + i];
        const b = data[partBase + pairCount + i];
        data[partBase + i] = (a * cos) - (b * sin);
        data[partBase + pairCount + i] = (b * cos) + (a * sin);
      }
    }
  }
}

async function applyVisionRopeToTensorInPlace(tensor, ropeCache, numTokens, numHeads, headDim, label) {
  const data = await readTensorF32(tensor, numTokens * numHeads * headDim, label);
  applyVisionRopeInPlace(data, ropeCache, numTokens, numHeads, headDim);
  uploadData(tensor.buffer, data, 0);
}

function applyRmsNormNoScaleInPlace(data, numTokens, hiddenSize, eps) {
  for (let tokenIdx = 0; tokenIdx < numTokens; tokenIdx++) {
    const base = tokenIdx * hiddenSize;
    let meanSquare = 0;
    for (let d = 0; d < hiddenSize; d++) {
      const value = data[base + d];
      meanSquare += value * value;
    }
    const inv = 1.0 / Math.sqrt(meanSquare / hiddenSize + eps);
    for (let d = 0; d < hiddenSize; d++) {
      data[base + d] *= inv;
    }
  }
}

function poolVisionHiddenStates(hiddenStates, gridHeight, gridWidth, hiddenSize, poolingKernelSize) {
  const pooledHeight = gridHeight / poolingKernelSize;
  const pooledWidth = gridWidth / poolingKernelSize;
  const outputLength = pooledHeight * pooledWidth;
  const output = new Float32Array(outputLength * hiddenSize);
  const scale = Math.sqrt(hiddenSize);
  const divisor = poolingKernelSize * poolingKernelSize;

  for (let pooledY = 0; pooledY < pooledHeight; pooledY++) {
    for (let pooledX = 0; pooledX < pooledWidth; pooledX++) {
      const outIdx = pooledY * pooledWidth + pooledX;
      const outBase = outIdx * hiddenSize;
      for (let ky = 0; ky < poolingKernelSize; ky++) {
        for (let kx = 0; kx < poolingKernelSize; kx++) {
          const srcPatch = ((pooledY * poolingKernelSize + ky) * gridWidth) + (pooledX * poolingKernelSize + kx);
          const srcBase = srcPatch * hiddenSize;
          for (let d = 0; d < hiddenSize; d++) {
            output[outBase + d] += hiddenStates[srcBase + d];
          }
        }
      }
      for (let d = 0; d < hiddenSize; d++) {
        output[outBase + d] = (output[outBase + d] / divisor) * scale;
      }
    }
  }

  return {
    output,
    outputLength,
  };
}

async function runVisionAttention(hiddenTensor, layerWeights, visionConfig, ropeCache, numTokens, hiddenSize) {
  const numHeads = Number(visionConfig.numHeads);
  const numKVHeads = Number(visionConfig.numKeyValueHeads ?? numHeads);
  const headDim = Number(visionConfig.headDim);

  let qTensor = null;
  let kTensor = null;
  let vTensor = null;
  let qNormTensor = null;
  let kNormTensor = null;
  let vNormTensor = null;
  let attnTensor = null;
  try {
    qTensor = await runClippableLinear(
      hiddenTensor,
      layerWeights.qProj,
      numTokens,
      numHeads * headDim,
      hiddenSize,
      layerWeights.qProjClip,
      'gemma4_vision_q_proj'
    );
    kTensor = await runClippableLinear(
      hiddenTensor,
      layerWeights.kProj,
      numTokens,
      numKVHeads * headDim,
      hiddenSize,
      layerWeights.kProjClip,
      'gemma4_vision_k_proj'
    );
    vTensor = await runClippableLinear(
      hiddenTensor,
      layerWeights.vProj,
      numTokens,
      numKVHeads * headDim,
      hiddenSize,
      layerWeights.vProjClip,
      'gemma4_vision_v_proj'
    );

    qNormTensor = await runRMSNorm(
      reshapeTensor(qTensor, [numTokens * numHeads, headDim], 'gemma4_vision_q_flat'),
      layerWeights.qNorm,
      visionConfig.eps,
      { batchSize: numTokens * numHeads, hiddenSize: headDim }
    );
    kNormTensor = await runRMSNorm(
      reshapeTensor(kTensor, [numTokens * numKVHeads, headDim], 'gemma4_vision_k_flat'),
      layerWeights.kNorm,
      visionConfig.eps,
      { batchSize: numTokens * numKVHeads, hiddenSize: headDim }
    );
    vNormTensor = await runRMSNorm(
      reshapeTensor(vTensor, [numTokens * numKVHeads, headDim], 'gemma4_vision_v_flat'),
      getQKNormOnesBuffer(headDim),
      visionConfig.eps,
      { batchSize: numTokens * numKVHeads, hiddenSize: headDim }
    );

    releaseBuffer(qTensor.buffer);
    releaseBuffer(kTensor.buffer);
    releaseBuffer(vTensor.buffer);
    qTensor = null;
    kTensor = null;
    vTensor = null;

    await applyVisionRopeToTensorInPlace(qNormTensor, ropeCache, numTokens, numHeads, headDim, 'gemma4_vision_q_rope');
    await applyVisionRopeToTensorInPlace(kNormTensor, ropeCache, numTokens, numKVHeads, headDim, 'gemma4_vision_k_rope');

    attnTensor = await runAttention(
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
      }
    );

    releaseBuffer(qNormTensor.buffer);
    releaseBuffer(kNormTensor.buffer);
    releaseBuffer(vNormTensor.buffer);
    qNormTensor = null;
    kNormTensor = null;
    vNormTensor = null;

    const output = await runClippableLinear(
      reshapeTensor(attnTensor, [numTokens, hiddenSize], 'gemma4_vision_attn_flat'),
      layerWeights.oProj,
      numTokens,
      hiddenSize,
      hiddenSize,
      layerWeights.oProjClip,
      'gemma4_vision_o_proj'
    );
    releaseBuffer(attnTensor.buffer);
    attnTensor = null;
    return output;
  } catch (error) {
    if (attnTensor) releaseBuffer(attnTensor.buffer);
    if (vNormTensor) releaseBuffer(vNormTensor.buffer);
    if (kNormTensor) releaseBuffer(kNormTensor.buffer);
    if (qNormTensor) releaseBuffer(qNormTensor.buffer);
    if (vTensor) releaseBuffer(vTensor.buffer);
    if (kTensor) releaseBuffer(kTensor.buffer);
    if (qTensor) releaseBuffer(qTensor.buffer);
    throw error;
  }
}

async function runVisionMlp(hiddenTensor, layerWeights, visionConfig, numTokens, hiddenSize) {
  const intermediateSize = Number(visionConfig.intermediateSize);
  let gateTensor = null;
  let upTensor = null;
  let activatedTensor = null;
  try {
    gateTensor = await runClippableLinear(
      hiddenTensor,
      layerWeights.gateProj,
      numTokens,
      intermediateSize,
      hiddenSize,
      layerWeights.gateProjClip,
      'gemma4_vision_gate_proj'
    );
    upTensor = await runClippableLinear(
      hiddenTensor,
      layerWeights.upProj,
      numTokens,
      intermediateSize,
      hiddenSize,
      layerWeights.upProjClip,
      'gemma4_vision_up_proj'
    );
    activatedTensor = await runGeLU(gateTensor, {
      size: numTokens * intermediateSize,
      gate: upTensor,
    });
    releaseBuffer(gateTensor.buffer);
    releaseBuffer(upTensor.buffer);
    gateTensor = null;
    upTensor = null;

    const output = await runClippableLinear(
      activatedTensor,
      layerWeights.downProj,
      numTokens,
      hiddenSize,
      intermediateSize,
      layerWeights.downProjClip,
      'gemma4_vision_down_proj'
    );
    releaseBuffer(activatedTensor.buffer);
    activatedTensor = null;
    return output;
  } catch (error) {
    if (activatedTensor) releaseBuffer(activatedTensor.buffer);
    if (upTensor) releaseBuffer(upTensor.buffer);
    if (gateTensor) releaseBuffer(gateTensor.buffer);
    throw error;
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
  const hiddenSize = Number(visionConfig.hiddenSize);
  const patchSize = Number(visionConfig.patchSize);
  const poolingKernelSize = Number(visionConfig.poolingKernelSize);
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

  const patchTensor = createTensorFromBuffer(
    acquireBuffer(scaledPatches.byteLength, undefined, 'gemma4_vision_patches'),
    [preprocessed.numPatches, 3 * patchSize * patchSize],
    'gemma4_vision_patches'
  );
  uploadData(patchTensor.buffer, scaledPatches, 0);

  const positionEmbeddings = buildPatchPositionEmbeddings(
    weights.patchPositionEmbeddingTable,
    preprocessed.positions,
    Number(visionConfig.positionEmbeddingSize),
    hiddenSize
  );
  const positionTensor = createTensorFromBuffer(
    acquireBuffer(positionEmbeddings.byteLength, undefined, 'gemma4_vision_position_embeddings'),
    [preprocessed.numPatches, hiddenSize],
    'gemma4_vision_position_embeddings'
  );
  uploadData(positionTensor.buffer, positionEmbeddings, 0);

  let hiddenTensor = null;
  try {
    hiddenTensor = await runMatmul(
      patchTensor,
      weights.patchInputProj,
      preprocessed.numPatches,
      hiddenSize,
      3 * patchSize * patchSize,
      { outputDtype: 'f32', transposeB: 'auto' }
    );
    releaseBuffer(patchTensor.buffer);

    const embedded = await runResidualAdd(hiddenTensor, positionTensor, preprocessed.numPatches * hiddenSize);
    releaseBuffer(hiddenTensor.buffer);
    releaseBuffer(positionTensor.buffer);
    hiddenTensor = reshapeTensor(embedded, [preprocessed.numPatches, hiddenSize], 'gemma4_vision_hidden_0');

    const ropeCache = buildVisionRopeCache(preprocessed.positions, Number(visionConfig.headDim), ropeTheta);

    for (let layerIdx = 0; layerIdx < weights.layers.length; layerIdx++) {
      const layerWeights = weights.layers[layerIdx];
      const inputNorm = await runRMSNorm(
        hiddenTensor,
        layerWeights.inputLayerNorm,
        visionConfig.eps,
        { batchSize: preprocessed.numPatches, hiddenSize }
      );

      const attnOut = await runVisionAttention(
        inputNorm,
        layerWeights,
        visionConfig,
        ropeCache,
        preprocessed.numPatches,
        hiddenSize
      );
      releaseBuffer(inputNorm.buffer);

      const postAttnNorm = await runRMSNorm(
        attnOut,
        layerWeights.postAttentionLayerNorm,
        visionConfig.eps,
        { batchSize: preprocessed.numPatches, hiddenSize }
      );
      releaseBuffer(attnOut.buffer);

      const attnResidual = await runResidualAdd(
        hiddenTensor,
        postAttnNorm,
        preprocessed.numPatches * hiddenSize
      );
      releaseBuffer(hiddenTensor.buffer);
      releaseBuffer(postAttnNorm.buffer);
      hiddenTensor = reshapeTensor(attnResidual, [preprocessed.numPatches, hiddenSize], `gemma4_vision_hidden_attn_${layerIdx}`);

      const preFfNorm = await runRMSNorm(
        hiddenTensor,
        layerWeights.preFeedforwardLayerNorm,
        visionConfig.eps,
        { batchSize: preprocessed.numPatches, hiddenSize }
      );
      const mlpOut = await runVisionMlp(preFfNorm, layerWeights, visionConfig, preprocessed.numPatches, hiddenSize);
      releaseBuffer(preFfNorm.buffer);

      const postFfNorm = await runRMSNorm(
        mlpOut,
        layerWeights.postFeedforwardLayerNorm,
        visionConfig.eps,
        { batchSize: preprocessed.numPatches, hiddenSize }
      );
      releaseBuffer(mlpOut.buffer);

      const ffResidual = await runResidualAdd(
        hiddenTensor,
        postFfNorm,
        preprocessed.numPatches * hiddenSize
      );
      releaseBuffer(hiddenTensor.buffer);
      releaseBuffer(postFfNorm.buffer);
      hiddenTensor = reshapeTensor(ffResidual, [preprocessed.numPatches, hiddenSize], `gemma4_vision_hidden_ff_${layerIdx}`);
    }

    const hiddenCpu = await readTensorF32(hiddenTensor, preprocessed.numPatches * hiddenSize, 'gemma4_vision_hidden_final');
    releaseBuffer(hiddenTensor.buffer);
    hiddenTensor = null;

    const pooled = poolVisionHiddenStates(
      hiddenCpu,
      preprocessed.gridHeight,
      preprocessed.gridWidth,
      hiddenSize,
      poolingKernelSize
    );
    applyRmsNormNoScaleInPlace(pooled.output, pooled.outputLength, hiddenSize, visionConfig.eps);

    const pooledTensor = createTensorFromBuffer(
      acquireBuffer(pooled.output.byteLength, undefined, 'gemma4_vision_pooled'),
      [pooled.outputLength, hiddenSize],
      'gemma4_vision_pooled'
    );
    uploadData(pooledTensor.buffer, pooled.output, 0);

    const projected = await runMatmul(
      pooledTensor,
      weights.projector,
      pooled.outputLength,
      weights.textHiddenSize,
      hiddenSize,
      { outputDtype: 'f32', transposeB: 'auto' }
    );
    releaseBuffer(pooledTensor.buffer);

    return {
      features: projected.buffer,
      numTokens: pooled.outputLength,
      gridThw: [1, preprocessed.gridHeight, preprocessed.gridWidth],
      imageWidth: preprocessed.gridWidth * patchSize,
      imageHeight: preprocessed.gridHeight * patchSize,
    };
  } catch (error) {
    if (hiddenTensor) releaseBuffer(hiddenTensor.buffer);
    throw error;
  }
}
