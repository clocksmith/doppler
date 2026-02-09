export function pixelShuffleRef(input, options) {
  const { outChannels, outHeight, outWidth, gridWidth, patchSize, patchChannels } = options;
  const output = new Float32Array(outChannels * outHeight * outWidth);
  const spatial = outHeight * outWidth;
  for (let idx = 0; idx < output.length; idx++) {
    const c = Math.floor(idx / spatial);
    const rem = idx - c * spatial;
    const y = Math.floor(rem / outWidth);
    const x = rem - y * outWidth;
    const gridY = Math.floor(y / patchSize);
    const gridX = Math.floor(x / patchSize);
    const subY = y - gridY * patchSize;
    const subX = x - gridX * patchSize;
    const tokenIdx = gridY * gridWidth + gridX;
    const patchIdx = (subY * patchSize + subX) * outChannels + c;
    const inputIdx = tokenIdx * patchChannels + patchIdx;
    output[idx] = input[inputIdx];
  }
  return output;
}
