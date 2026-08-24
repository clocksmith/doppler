
export function sampleFrames(frames, maxFrames) {
  if (!Array.isArray(frames) || frames.length === 0) {
    throw new Error('[Video] sampleFrames requires a non-empty array of decoded frames.');
  }
  if (!Number.isFinite(maxFrames) || maxFrames < 1) {
    throw new Error(`[Video] maxFrames must be a positive integer, got ${maxFrames}.`);
  }

  const totalFrames = frames.length;
  const numSampled = Math.min(totalFrames, Math.trunc(maxFrames));

  if (numSampled >= totalFrames) {
    return frames.map((frame, i) => ({ ...frame, frameIndex: i }));
  }

  const sampled = [];
  for (let i = 0; i < numSampled; i++) {
    const idx = Math.round((i * (totalFrames - 1)) / (numSampled - 1));
    sampled.push({ ...frames[idx], frameIndex: idx });
  }
  return sampled;
}
