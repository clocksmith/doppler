export function upsample2dRef(input, options) {
  const { channels, height, width, inHeight, inWidth, outHeight, outWidth, scale } = options;
  const srcHeight = Number.isFinite(height) ? height : inHeight;
  const srcWidth = Number.isFinite(width) ? width : inWidth;
  const output = new Float32Array(channels * outHeight * outWidth);
  for (let c = 0; c < channels; c++) {
    for (let oy = 0; oy < outHeight; oy++) {
      for (let ox = 0; ox < outWidth; ox++) {
        const inY = Math.floor(oy / scale);
        const inX = Math.floor(ox / scale);
        const inIdx = (c * srcHeight + inY) * srcWidth + inX;
        const outIdx = (c * outHeight + oy) * outWidth + ox;
        output[outIdx] = input[inIdx];
      }
    }
  }
  return output;
}

export function upsample2dBackwardRef(gradOutput, options) {
  const { channels, inHeight, inWidth, outHeight, outWidth, scale } = options;
  const gradInput = new Float32Array(channels * inHeight * inWidth);
  for (let c = 0; c < channels; c++) {
    for (let oy = 0; oy < outHeight; oy++) {
      for (let ox = 0; ox < outWidth; ox++) {
        const inY = Math.floor(oy / scale);
        const inX = Math.floor(ox / scale);
        if (inY < inHeight && inX < inWidth) {
          const inIdx = (c * inHeight + inY) * inWidth + inX;
          const outIdx = (c * outHeight + oy) * outWidth + ox;
          gradInput[inIdx] += gradOutput[outIdx];
        }
      }
    }
  }
  return gradInput;
}
