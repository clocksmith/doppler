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
