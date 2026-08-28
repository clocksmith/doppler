import assert from 'node:assert/strict';

import {
  preprocessGlmOcrImage,
  resizeGlmOcrImageBicubic,
  resolveGlmOcrImageSize,
} from '../../src/inference/pipelines/vision/glmocr.js';

const visionConfig = {
  patchSize: 1,
  spatialMergeSize: 2,
  temporalPatchSize: 2,
  inChannels: 3,
  minPixels: 8,
  maxPixels: 8,
  normalization: {
    mean: [0, 0, 0],
    std: [1, 1, 1],
  },
};

assert.deepEqual(resolveGlmOcrImageSize(2, 2, visionConfig), {
  targetWidth: 2,
  targetHeight: 2,
});

const pixels = new Uint8Array([
  255, 0, 0,
  0, 255, 0,
  0, 0, 255,
  255, 255, 255,
]);
const result = preprocessGlmOcrImage(pixels, 2, 2, visionConfig);
assert.equal(result.numPatches, 4);
assert.equal(result.patchDim, 6);
assert.equal(result.gridHeight, 2);
assert.equal(result.gridWidth, 2);

const expected = new Float32Array([
  1, 1, 0, 0, 0, 0,
  0, 0, 1, 1, 0, 0,
  0, 0, 0, 0, 1, 1,
  1, 1, 1, 1, 1, 1,
]);
assert.deepEqual(result.patches, expected);

function interleaveRgb(red, green, blue) {
  assert.equal(red.length, green.length);
  assert.equal(red.length, blue.length);
  const result = new Uint8Array(red.length * 3);
  for (let index = 0; index < red.length; index++) {
    result[index * 3] = red[index];
    result[index * 3 + 1] = green[index];
    result[index * 3 + 2] = blue[index];
  }
  return result;
}

const compactSource = interleaveRgb(
  [
    0, 32, 64, 96, 128, 160, 192,
    17, 49, 81, 113, 145, 177, 209,
    34, 66, 98, 130, 162, 194, 226,
    51, 83, 115, 147, 179, 211, 243,
    68, 100, 132, 164, 196, 228, 255,
  ],
  [
    255, 224, 192, 160, 128, 96, 64,
    238, 207, 175, 143, 111, 79, 47,
    221, 190, 158, 126, 94, 62, 30,
    204, 173, 141, 109, 77, 45, 13,
    187, 156, 124, 92, 60, 28, 0,
  ],
  [
    13, 29, 47, 71, 101, 137, 179,
    19, 37, 59, 87, 121, 161, 207,
    31, 53, 79, 109, 145, 187, 233,
    43, 67, 95, 127, 165, 209, 255,
    59, 83, 113, 149, 191, 239, 251,
  ]
);
const compactExpected = interleaveRgb(
  [
    5, 41, 79, 117, 155, 191,
    26, 62, 100, 138, 176, 212,
    48, 84, 122, 160, 198, 234,
    69, 105, 143, 181, 219, 251,
  ],
  [
    250, 215, 177, 139, 101, 65,
    229, 194, 156, 118, 80, 44,
    207, 172, 134, 96, 58, 22,
    186, 151, 113, 75, 37, 4,
  ],
  [
    14, 34, 58, 90, 131, 179,
    25, 48, 78, 115, 161, 213,
    41, 68, 102, 142, 190, 244,
    59, 87, 124, 169, 224, 252,
  ]
);
assert.deepEqual(
  resizeGlmOcrImageBicubic(compactSource, 7, 5, 6, 4),
  compactExpected,
  'GLM-OCR resize must match the pinned Torchvision 2.13 uint8 bicubic-antialias oracle'
);

assert.throws(
  () => resolveGlmOcrImageSize(1, 500, visionConfig),
  /aspect ratio/
);

console.log('glmocr-image-preprocess-contract.test: ok');
