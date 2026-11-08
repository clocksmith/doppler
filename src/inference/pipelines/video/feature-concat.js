// Plan the copies that concatenate per-frame vision features into one buffer.
// Each frame result declares its own feature width; every frame must agree.
/** @param {ReadonlyArray<{ featureSize?: unknown, numTokens?: unknown }>} frameResults */
export function planVideoFeatureConcat(frameResults) {
  if (!Array.isArray(frameResults) || frameResults.length === 0) {
    throw new Error('[Video] No frames were encoded.');
  }
  const featureSize = frameResults[0]?.featureSize;
  if (typeof featureSize !== 'number' || !Number.isSafeInteger(featureSize) || featureSize <= 0) {
    throw new Error(`[Video] Frame 0 must declare a positive integer featureSize; got ${featureSize}.`);
  }
  const bytesPerToken = featureSize * Float32Array.BYTES_PER_ELEMENT;
  /** @type {Array<{ offset: number, byteLength: number }>} */
  const copies = [];
  let totalTokens = 0;
  frameResults.forEach((result, frame) => {
    if (result?.featureSize !== featureSize) {
      throw new Error(
        `[Video] Frame ${frame} declares featureSize ${result?.featureSize}; frame 0 declares ${featureSize}.`
      );
    }
    const numTokens = result.numTokens;
    if (typeof numTokens !== 'number' || !Number.isSafeInteger(numTokens) || numTokens <= 0) {
      throw new Error(`[Video] Frame ${frame} must produce a positive integer numTokens; got ${numTokens}.`);
    }
    copies.push({ offset: totalTokens * bytesPerToken, byteLength: numTokens * bytesPerToken });
    totalTokens += numTokens;
  });
  return { featureSize, totalTokens, totalBytes: totalTokens * bytesPerToken, copies };
}
