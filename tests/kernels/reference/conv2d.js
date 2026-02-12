export function conv2dRef(input, weight, bias, options) {
  const { inChannels, outChannels, height, width, kernelH, kernelW, stride = 1, pad = 0 } = options;
  const outHeight = Math.floor((height + pad * 2 - kernelH) / stride) + 1;
  const outWidth = Math.floor((width + pad * 2 - kernelW) / stride) + 1;
  const output = new Float32Array(outChannels * outHeight * outWidth);
  for (let oc = 0; oc < outChannels; oc++) {
    for (let oy = 0; oy < outHeight; oy++) {
      for (let ox = 0; ox < outWidth; ox++) {
        let sum = bias ? bias[oc] : 0;
        for (let ic = 0; ic < inChannels; ic++) {
          for (let ky = 0; ky < kernelH; ky++) {
            const inY = oy * stride + ky - pad;
            if (inY < 0 || inY >= height) continue;
            for (let kx = 0; kx < kernelW; kx++) {
              const inX = ox * stride + kx - pad;
              if (inX < 0 || inX >= width) continue;
              const inputIdx = (ic * height + inY) * width + inX;
              const weightIdx = (((oc * inChannels + ic) * kernelH + ky) * kernelW + kx);
              sum += input[inputIdx] * weight[weightIdx];
            }
          }
        }
        output[(oc * outHeight + oy) * outWidth + ox] = sum;
      }
    }
  }
  return output;
}

export function conv2dBackwardRef(input, weight, gradOutput, options) {
  const { inChannels, outChannels, height, width, outHeight, outWidth, kernelH, kernelW, stride, pad } = options;
  const gradInput = new Float32Array(inChannels * height * width);
  const gradWeight = new Float32Array(outChannels * inChannels * kernelH * kernelW);

  for (let oc = 0; oc < outChannels; oc++) {
    for (let ic = 0; ic < inChannels; ic++) {
      for (let ky = 0; ky < kernelH; ky++) {
        for (let kx = 0; kx < kernelW; kx++) {
          for (let oy = 0; oy < outHeight; oy++) {
            for (let ox = 0; ox < outWidth; ox++) {
              const iy = oy * stride + ky - pad;
              const ix = ox * stride + kx - pad;
              if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                const dy = gradOutput[(oc * outHeight + oy) * outWidth + ox];
                const w = weight[(((oc * inChannels + ic) * kernelH + ky) * kernelW + kx)];
                gradInput[(ic * height + iy) * width + ix] += dy * w;

                const x = input[(ic * height + iy) * width + ix];
                gradWeight[(((oc * inChannels + ic) * kernelH + ky) * kernelW + kx)] += dy * x;
              }
            }
          }
        }
      }
    }
  }
  return { gradInput, gradWeight };
}
