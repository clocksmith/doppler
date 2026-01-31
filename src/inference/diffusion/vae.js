function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

export function decodeLatents(latents, options) {
  const width = options?.width ?? 512;
  const height = options?.height ?? 512;
  const latentWidth = options?.latentWidth ?? Math.max(1, Math.floor(width / 8));
  const latentHeight = options?.latentHeight ?? Math.max(1, Math.floor(height / 8));
  const channels = options?.latentChannels ?? 4;
  const scale = options?.latentScale ?? 8;
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
