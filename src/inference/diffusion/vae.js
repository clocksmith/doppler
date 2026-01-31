function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

export function decodeLatents(latents, options) {
  if (!options) {
    throw new Error('decodeLatents requires options');
  }
  const width = options.width;
  const height = options.height;
  const scale = options.latentScale;
  if (!Number.isFinite(width) || !Number.isFinite(height)) {
    throw new Error('decodeLatents requires width/height');
  }
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new Error('decodeLatents requires latentScale');
  }
  const latentWidth = Number.isFinite(options.latentWidth)
    ? options.latentWidth
    : Math.max(1, Math.floor(width / scale));
  const latentHeight = Number.isFinite(options.latentHeight)
    ? options.latentHeight
    : Math.max(1, Math.floor(height / scale));
  const channels = Number.isFinite(options.latentChannels)
    ? options.latentChannels
    : 0;
  if (!Number.isFinite(channels) || channels <= 0) {
    throw new Error('decodeLatents requires latentChannels');
  }
  const output = new Uint8ClampedArray(width * height * 4);

  for (let y = 0; y < height; y++) {
    const ly = Math.min(latentHeight - 1, Math.floor(y / scale));
    for (let x = 0; x < width; x++) {
      const lx = Math.min(latentWidth - 1, Math.floor(x / scale));
      const latentIndex = (ly * latentWidth + lx) * channels;
      const r = latents[latentIndex] ?? 0;
      const g = latents[latentIndex + 1] ?? r;
      const b = latents[latentIndex + 2] ?? r;
      const outIndex = (y * width + x) * 4;
      output[outIndex] = clamp(Math.round((r * 0.5 + 0.5) * 255), 0, 255);
      output[outIndex + 1] = clamp(Math.round((g * 0.5 + 0.5) * 255), 0, 255);
      output[outIndex + 2] = clamp(Math.round((b * 0.5 + 0.5) * 255), 0, 255);
      output[outIndex + 3] = 255;
    }
  }

  return output;
}
