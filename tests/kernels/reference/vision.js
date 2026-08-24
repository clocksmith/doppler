export function visionPatchEmbedRef(image, weight, bias, geometry) {
  const {
    gridHeight,
    gridWidth,
    channels,
    patchSize,
    temporalPatchSize,
    hiddenSize,
  } = geometry;
  const imageHeight = gridHeight * patchSize;
  const imageWidth = gridWidth * patchSize;
  const spatialPatchArea = channels * patchSize * patchSize;
  const temporalPatchArea = temporalPatchSize * spatialPatchArea;
  const output = new Float32Array(gridHeight * gridWidth * hiddenSize);
  for (let patchY = 0; patchY < gridHeight; patchY++) {
    for (let patchX = 0; patchX < gridWidth; patchX++) {
      const patchIndex = patchY * gridWidth + patchX;
      for (let hiddenIndex = 0; hiddenIndex < hiddenSize; hiddenIndex++) {
        let value = bias ? bias[hiddenIndex] : 0;
        for (let channel = 0; channel < channels; channel++) {
          for (let localY = 0; localY < patchSize; localY++) {
            for (let localX = 0; localX < patchSize; localX++) {
              const imageY = patchY * patchSize + localY;
              const imageX = patchX * patchSize + localX;
              const imageIndex = channel * imageHeight * imageWidth + imageY * imageWidth + imageX;
              const spatialIndex = channel * patchSize * patchSize + localY * patchSize + localX;
              for (let temporal = 0; temporal < temporalPatchSize; temporal++) {
                const weightIndex = hiddenIndex * temporalPatchArea + temporal * spatialPatchArea + spatialIndex;
                value += image[imageIndex] * weight[weightIndex];
              }
            }
          }
        }
        output[patchIndex * hiddenSize + hiddenIndex] = value;
      }
    }
  }
  return output;
}

export function visionSpatialMergeRef(input, geometry) {
  const { gridHeight, gridWidth, hiddenSize, mergeSize } = geometry;
  const mergedHeight = gridHeight / mergeSize;
  const mergedWidth = gridWidth / mergeSize;
  const concatDim = mergeSize * mergeSize * hiddenSize;
  const output = new Float32Array(mergedHeight * mergedWidth * concatDim);
  for (let mergedY = 0; mergedY < mergedHeight; mergedY++) {
    for (let mergedX = 0; mergedX < mergedWidth; mergedX++) {
      const mergedIndex = mergedY * mergedWidth + mergedX;
      for (let localY = 0; localY < mergeSize; localY++) {
        for (let localX = 0; localX < mergeSize; localX++) {
          const patchInMerge = localY * mergeSize + localX;
          const sourceY = mergedY * mergeSize + localY;
          const sourceX = mergedX * mergeSize + localX;
          const sourcePatch = sourceY * gridWidth + sourceX;
          for (let hiddenIndex = 0; hiddenIndex < hiddenSize; hiddenIndex++) {
            const outputIndex = mergedIndex * concatDim + patchInMerge * hiddenSize + hiddenIndex;
            output[outputIndex] = input[sourcePatch * hiddenSize + hiddenIndex];
          }
        }
      }
    }
  }
  return output;
}

export function visionPositionEmbeddingRef(table, geometry) {
  const { gridHeight, gridWidth, positionEmbeddingSize, hiddenSize } = geometry;
  const output = new Float32Array(gridHeight * gridWidth * hiddenSize);
  const yTableOffset = positionEmbeddingSize * hiddenSize;
  for (let y = 0; y < gridHeight; y++) {
    for (let x = 0; x < gridWidth; x++) {
      const tokenIndex = y * gridWidth + x;
      for (let hiddenIndex = 0; hiddenIndex < hiddenSize; hiddenIndex++) {
        output[tokenIndex * hiddenSize + hiddenIndex] = table[x * hiddenSize + hiddenIndex]
          + table[yTableOffset + y * hiddenSize + hiddenIndex];
      }
    }
  }
  return output;
}

export function visionRope2DRef(input, geometry) {
  const { numTokens, numHeads, headDim, gridWidth, ropeTheta } = geometry;
  const output = new Float32Array(input);
  const spatialDim = headDim / 2;
  const pairsPerAxis = spatialDim / 2;
  for (let tokenIndex = 0; tokenIndex < numTokens; tokenIndex++) {
    const positions = [tokenIndex % gridWidth, Math.floor(tokenIndex / gridWidth)];
    for (let headIndex = 0; headIndex < numHeads; headIndex++) {
      const headBase = tokenIndex * numHeads * headDim + headIndex * headDim;
      for (let axis = 0; axis < 2; axis++) {
        const axisBase = headBase + axis * spatialDim;
        for (let pairIndex = 0; pairIndex < pairsPerAxis; pairIndex++) {
          const angle = positions[axis] / (ropeTheta ** ((2 * pairIndex) / spatialDim));
          const cosine = Math.cos(angle);
          const sine = Math.sin(angle);
          const firstIndex = axisBase + pairIndex;
          const secondIndex = firstIndex + pairsPerAxis;
          const first = output[firstIndex];
          const second = output[secondIndex];
          output[firstIndex] = first * cosine - second * sine;
          output[secondIndex] = second * cosine + first * sine;
        }
      }
    }
  }
  return output;
}

export function visionAveragePoolRef(input, geometry) {
  const { gridHeight, gridWidth, hiddenSize, poolingSize } = geometry;
  const pooledHeight = gridHeight / poolingSize;
  const pooledWidth = gridWidth / poolingSize;
  const output = new Float32Array(pooledHeight * pooledWidth * hiddenSize);
  const scale = Math.sqrt(hiddenSize) / (poolingSize * poolingSize);
  for (let pooledY = 0; pooledY < pooledHeight; pooledY++) {
    for (let pooledX = 0; pooledX < pooledWidth; pooledX++) {
      const pooledIndex = pooledY * pooledWidth + pooledX;
      for (let hiddenIndex = 0; hiddenIndex < hiddenSize; hiddenIndex++) {
        let sum = 0;
        for (let localY = 0; localY < poolingSize; localY++) {
          for (let localX = 0; localX < poolingSize; localX++) {
            const sourceY = pooledY * poolingSize + localY;
            const sourceX = pooledX * poolingSize + localX;
            sum += input[(sourceY * gridWidth + sourceX) * hiddenSize + hiddenIndex];
          }
        }
        output[pooledIndex * hiddenSize + hiddenIndex] = sum * scale;
      }
    }
  }
  return output;
}
