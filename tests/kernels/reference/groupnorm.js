export function groupNormRef(data, weight, bias, channels, height, width, numGroups, eps) {
  const output = new Float32Array(channels * height * width);
  const channelsPerGroup = Math.floor(channels / numGroups);
  const spatial = height * width;
  for (let g = 0; g < numGroups; g++) {
    const cStart = g * channelsPerGroup;
    const cEnd = cStart + channelsPerGroup;
    let mean = 0;
    let count = 0;
    for (let c = cStart; c < cEnd; c++) {
      const base = c * spatial;
      for (let i = 0; i < spatial; i++) {
        mean += data[base + i];
        count++;
      }
    }
    mean /= count;
    let varSum = 0;
    for (let c = cStart; c < cEnd; c++) {
      const base = c * spatial;
      for (let i = 0; i < spatial; i++) {
        const diff = data[base + i] - mean;
        varSum += diff * diff;
      }
    }
    const invStd = 1 / Math.sqrt(varSum / count + eps);
    for (let c = cStart; c < cEnd; c++) {
      const base = c * spatial;
      for (let i = 0; i < spatial; i++) {
        const norm = (data[base + i] - mean) * invStd;
        output[base + i] = norm * weight[c] + bias[c];
      }
    }
  }
  return output;
}
