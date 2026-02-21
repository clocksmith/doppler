#!/usr/bin/env node

const BITS_VIEW = new DataView(new ArrayBuffer(4));

function float32ToBits(value) {
  BITS_VIEW.setFloat32(0, value, true);
  return BITS_VIEW.getUint32(0, true);
}

function bitsToFloat32(value) {
  BITS_VIEW.setUint32(0, value >>> 0, true);
  return BITS_VIEW.getFloat32(0, true);
}

function float32ToBFloat16(value) {
  const bits = float32ToBits(value);
  const lsb = (bits >> 16) & 1;
  const roundingBias = 0x7fff + lsb;
  return ((bits + roundingBias) >> 16) & 0xffff;
}

function bfloat16ToFloat32(value) {
  return bitsToFloat32((value & 0xffff) << 16);
}

function roundToBF16(value) {
  return bfloat16ToFloat32(float32ToBFloat16(value));
}

function roundToF16(value) {
  return float16ToFloat32(float32ToFloat16(value));
}

function float32ToFloat16(value) {
  const floatView = new Float32Array(1);
  const int32View = new Int32Array(floatView.buffer);
  floatView[0] = value;
  const bits = int32View[0];

  const sign = (bits >> 31) & 0x1;
  let exp = (bits >> 23) & 0xff;
  let frac = bits & 0x7fffff;

  if (exp === 0xff) {
    return (sign << 15) | 0x7c00 | (frac ? 0x200 : 0);
  }

  if (exp === 0) {
    return sign << 15;
  }

  exp = exp - 127 + 15;

  if (exp >= 31) {
    return (sign << 15) | 0x7c00;
  }

  if (exp <= 0) {
    if (exp < -10) {
      return sign << 15;
    }
    frac = (frac | 0x800000) >> (1 - exp);
    return (sign << 15) | (frac >> 13);
  }

  return (sign << 15) | (exp << 10) | (frac >> 13);
}

function float16ToFloat32(value) {
  const sign = (value >> 15) & 0x1;
  const exp = (value >> 10) & 0x1f;
  const frac = value & 0x3ff;

  if (exp === 0) {
    if (frac === 0) {
      return sign ? -0 : 0;
    }
    return (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024);
  }

  if (exp === 31) {
    return frac ? NaN : (sign ? -Infinity : Infinity);
  }

  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

function createRng(seed = 0xdecafbad) {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function gaussian(rng) {
  const u1 = Math.max(rng(), Number.EPSILON);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function dotF32(a, b) {
  let acc = 0;
  for (let i = 0; i < a.length; i += 1) {
    acc += a[i] * b[i];
  }
  return acc;
}

function dotBF16Strict(a, b) {
  let acc = 0;
  for (let i = 0; i < a.length; i += 1) {
    const aa = roundToBF16(a[i]);
    const bb = roundToBF16(b[i]);
    const product = roundToBF16(aa * bb);
    acc = roundToBF16(acc + product);
  }
  return acc;
}

function dotF16Strict(a, b) {
  let acc = 0;
  for (let i = 0; i < a.length; i += 1) {
    const aa = roundToF16(a[i]);
    const bb = roundToF16(b[i]);
    const product = roundToF16(aa * bb);
    acc = roundToF16(acc + product);
  }
  return acc;
}

function dotBF16F32Acc(a, b) {
  let acc = 0;
  for (let i = 0; i < a.length; i += 1) {
    acc += roundToBF16(a[i]) * roundToBF16(b[i]);
  }
  return acc;
}

function dotF16F32Acc(a, b) {
  let acc = 0;
  for (let i = 0; i < a.length; i += 1) {
    acc += roundToF16(a[i]) * roundToF16(b[i]);
  }
  return acc;
}

function runDotStudy({
  trials = 300,
  width = 2048,
  scale = 0.25,
  seed = 0xc0ffee,
} = {}) {
  const rng = createRng(seed);
  let errBf16Strict = 0;
  let errBf16F32Acc = 0;
  let errF16Strict = 0;
  let errF16F32Acc = 0;

  for (let trial = 0; trial < trials; trial += 1) {
    const a = new Float32Array(width);
    const b = new Float32Array(width);
    for (let i = 0; i < width; i += 1) {
      a[i] = gaussian(rng) * scale;
      b[i] = gaussian(rng) * scale;
    }

    const baseline = dotF32(a, b);
    errBf16Strict += Math.abs(dotBF16Strict(a, b) - baseline);
    errBf16F32Acc += Math.abs(dotBF16F32Acc(a, b) - baseline);
    errF16Strict += Math.abs(dotF16Strict(a, b) - baseline);
    errF16F32Acc += Math.abs(dotF16F32Acc(a, b) - baseline);
  }

  return {
    trials,
    width,
    meanAbsErr: {
      bf16Strict: errBf16Strict / trials,
      bf16InputF32Acc: errBf16F32Acc / trials,
      f16Strict: errF16Strict / trials,
      f16InputF32Acc: errF16F32Acc / trials,
    },
  };
}

function runDeferredRoundingStudy({
  steps = 4096,
  blockSizes = [1, 2, 4, 8, 16],
  seed = 0x12345678,
} = {}) {
  const rng = createRng(seed);
  const deltas = new Float32Array(steps);
  for (let i = 0; i < steps; i += 1) {
    deltas[i] = gaussian(rng) * 0.03;
  }

  let baseline = 0;
  for (let i = 0; i < steps; i += 1) {
    baseline += deltas[i];
  }

  const byBlock = {};
  for (const blockSize of blockSizes) {
    let state = roundToBF16(0);
    let pending = 0;
    let meanAbsErr = 0;
    let runningBaseline = 0;

    for (let i = 0; i < steps; i += 1) {
      pending += deltas[i];
      runningBaseline += deltas[i];
      if ((i + 1) % blockSize === 0) {
        state = roundToBF16(state + pending);
        pending = 0;
      }

      meanAbsErr += Math.abs(state - runningBaseline);
    }

    if (pending !== 0) {
      state = roundToBF16(state + pending);
    }

    byBlock[`roundEvery${blockSize}`] = {
      finalAbsErr: Math.abs(state - baseline),
      meanAbsErr: meanAbsErr / steps,
    };
  }

  return {
    steps,
    baselineFinal: baseline,
    byBlock,
  };
}

function findOverflowPoint({
  start = 1,
  growth = 1.13,
  maxSteps = 1024,
} = {}) {
  let value = start;
  let f16Step = null;
  let bf16Step = null;

  for (let step = 0; step < maxSteps; step += 1) {
    value *= growth;
    if (f16Step === null && !Number.isFinite(roundToF16(value))) {
      f16Step = step;
    }
    if (bf16Step === null && !Number.isFinite(roundToBF16(value))) {
      bf16Step = step;
    }
    if (f16Step !== null && bf16Step !== null) {
      break;
    }
  }

  return {
    start,
    growth,
    maxSteps,
    overflowStep: {
      f16: f16Step,
      bf16: bf16Step,
    },
  };
}

function summarizeVectorError(candidate, baseline) {
  let absErr = 0;
  let finiteCount = 0;
  let nonFiniteCount = 0;

  for (let i = 0; i < candidate.length; i += 1) {
    const value = candidate[i];
    if (!Number.isFinite(value)) {
      nonFiniteCount += 1;
      continue;
    }
    absErr += Math.abs(value - baseline[i]);
    finiteCount += 1;
  }

  return {
    meanAbsErr: finiteCount > 0 ? absErr / finiteCount : null,
    nonFiniteCount,
  };
}

function sumVector(values) {
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    sum += values[i];
  }
  return sum;
}

function rmsNormF32(values, eps = 1e-6) {
  let sumSq = 0;
  for (let i = 0; i < values.length; i += 1) {
    const value = values[i];
    sumSq += value * value;
  }
  const inv = 1 / Math.sqrt(sumSq / values.length + eps);
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i += 1) {
    out[i] = values[i] * inv;
  }
  return out;
}

function rmsNormQuantized(values, roundFn, {
  strict = false,
  roundOutput = true,
  eps = 1e-6,
} = {}) {
  const quantized = new Float32Array(values.length);
  for (let i = 0; i < values.length; i += 1) {
    quantized[i] = roundFn(values[i]);
  }

  let meanSq;
  if (strict) {
    let sumSq = roundFn(0);
    for (let i = 0; i < quantized.length; i += 1) {
      const sq = roundFn(quantized[i] * quantized[i]);
      sumSq = roundFn(sumSq + sq);
    }
    meanSq = roundFn(sumSq / quantized.length);
  } else {
    let sumSq = 0;
    for (let i = 0; i < quantized.length; i += 1) {
      sumSq += quantized[i] * quantized[i];
    }
    meanSq = sumSq / quantized.length;
  }

  const denomInput = strict
    ? roundFn(meanSq + eps)
    : (meanSq + eps);
  const inv = strict
    ? roundFn(1 / Math.sqrt(denomInput))
    : (1 / Math.sqrt(denomInput));

  const out = new Float32Array(values.length);
  for (let i = 0; i < quantized.length; i += 1) {
    const value = quantized[i] * inv;
    out[i] = roundOutput ? roundFn(value) : value;
  }
  return out;
}

function runRmsNormStudy({
  trials = 120,
  width = 4096,
  scales = [1, 4, 8, 16, 32, 64, 96, 128],
  eps = 1e-6,
  seed = 0x6b8b4567,
} = {}) {
  const rng = createRng(seed);
  const methods = {
    bf16Strict: (values) => rmsNormQuantized(values, roundToBF16, { strict: true, roundOutput: true, eps }),
    bf16InputF32Acc: (values) => rmsNormQuantized(values, roundToBF16, { strict: false, roundOutput: false, eps }),
    bf16InputF32AccRoundOut: (values) => rmsNormQuantized(values, roundToBF16, { strict: false, roundOutput: true, eps }),
    f16Strict: (values) => rmsNormQuantized(values, roundToF16, { strict: true, roundOutput: true, eps }),
    f16InputF32Acc: (values) => rmsNormQuantized(values, roundToF16, { strict: false, roundOutput: false, eps }),
    f16InputF32AccRoundOut: (values) => rmsNormQuantized(values, roundToF16, { strict: false, roundOutput: true, eps }),
  };

  const byScale = {};
  for (const scale of scales) {
    const totals = {};
    for (const key of Object.keys(methods)) {
      totals[key] = {
        meanAbsErrSum: 0,
        errSamples: 0,
        nonFiniteElements: 0,
        nonFiniteVectors: 0,
      };
    }

    for (let trial = 0; trial < trials; trial += 1) {
      const values = new Float32Array(width);
      for (let i = 0; i < width; i += 1) {
        values[i] = gaussian(rng) * scale;
      }

      const baseline = rmsNormF32(values, eps);
      for (const [name, runMethod] of Object.entries(methods)) {
        const candidate = runMethod(values);
        const stats = summarizeVectorError(candidate, baseline);
        const agg = totals[name];

        if (stats.meanAbsErr !== null) {
          agg.meanAbsErrSum += stats.meanAbsErr;
          agg.errSamples += 1;
        }
        agg.nonFiniteElements += stats.nonFiniteCount;
        if (stats.nonFiniteCount > 0) {
          agg.nonFiniteVectors += 1;
        }
      }
    }

    const scaleResult = {};
    for (const [name, agg] of Object.entries(totals)) {
      scaleResult[name] = {
        meanAbsErr: agg.errSamples > 0 ? agg.meanAbsErrSum / agg.errSamples : null,
        nonFiniteVectorRate: agg.nonFiniteVectors / trials,
        nonFiniteElementRate: agg.nonFiniteElements / (trials * width),
      };
    }
    byScale[`scale${scale}`] = scaleResult;
  }

  return {
    trials,
    width,
    eps,
    scales,
    byScale,
  };
}

function softmaxF32Stable(logits) {
  let maxLogit = -Infinity;
  for (let i = 0; i < logits.length; i += 1) {
    if (logits[i] > maxLogit) {
      maxLogit = logits[i];
    }
  }

  const exps = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i += 1) {
    const value = Math.exp(logits[i] - maxLogit);
    exps[i] = value;
    sum += value;
  }

  const out = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i += 1) {
    out[i] = exps[i] / sum;
  }
  return out;
}

function softmaxQuantized(logits, roundFn, {
  strict = false,
  subtractMax = true,
  roundOutput = true,
} = {}) {
  const quantized = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i += 1) {
    quantized[i] = roundFn(logits[i]);
  }

  let maxLogit = -Infinity;
  if (subtractMax) {
    for (let i = 0; i < quantized.length; i += 1) {
      if (quantized[i] > maxLogit) {
        maxLogit = quantized[i];
      }
    }
  }

  const exps = new Float32Array(logits.length);
  let sum = strict ? roundFn(0) : 0;
  let expOverflowCount = 0;
  for (let i = 0; i < quantized.length; i += 1) {
    const shifted = subtractMax
      ? (strict ? roundFn(quantized[i] - maxLogit) : (quantized[i] - maxLogit))
      : quantized[i];
    let expValue = Math.exp(shifted);
    if (strict) {
      expValue = roundFn(expValue);
    }
    if (!Number.isFinite(expValue)) {
      expOverflowCount += 1;
    }
    exps[i] = expValue;
    sum = strict ? roundFn(sum + expValue) : (sum + expValue);
  }

  const out = new Float32Array(logits.length);
  let nonFiniteCount = 0;
  for (let i = 0; i < exps.length; i += 1) {
    let value = exps[i] / sum;
    if (strict || roundOutput) {
      value = roundFn(value);
    }
    out[i] = value;
    if (!Number.isFinite(value)) {
      nonFiniteCount += 1;
    }
  }

  return {
    values: out,
    expOverflowCount,
    nonFiniteCount,
  };
}

function runSoftmaxStudy({
  trials = 250,
  width = 256,
  scales = [1, 4, 8, 12, 16, 20],
  seed = 0x4d595df4,
} = {}) {
  const rng = createRng(seed);
  const methods = {
    bf16StrictNaive: (values) => softmaxQuantized(values, roundToBF16, { strict: true, subtractMax: false, roundOutput: true }),
    bf16StrictStable: (values) => softmaxQuantized(values, roundToBF16, { strict: true, subtractMax: true, roundOutput: true }),
    bf16InputF32AccStable: (values) => softmaxQuantized(values, roundToBF16, { strict: false, subtractMax: true, roundOutput: false }),
    f16StrictNaive: (values) => softmaxQuantized(values, roundToF16, { strict: true, subtractMax: false, roundOutput: true }),
    f16StrictStable: (values) => softmaxQuantized(values, roundToF16, { strict: true, subtractMax: true, roundOutput: true }),
    f16InputF32AccStable: (values) => softmaxQuantized(values, roundToF16, { strict: false, subtractMax: true, roundOutput: false }),
  };

  const byScale = {};
  for (const scale of scales) {
    const totals = {};
    for (const key of Object.keys(methods)) {
      totals[key] = {
        meanAbsErrSum: 0,
        errSamples: 0,
        nonFiniteVectors: 0,
        expOverflowEvents: 0,
        probSumAbsErr: 0,
        probSumSamples: 0,
      };
    }

    for (let trial = 0; trial < trials; trial += 1) {
      const logits = new Float32Array(width);
      for (let i = 0; i < width; i += 1) {
        logits[i] = gaussian(rng) * scale;
      }

      const baseline = softmaxF32Stable(logits);
      for (const [name, runMethod] of Object.entries(methods)) {
        const candidate = runMethod(logits);
        const stats = summarizeVectorError(candidate.values, baseline);
        const agg = totals[name];

        if (stats.meanAbsErr !== null) {
          agg.meanAbsErrSum += stats.meanAbsErr;
          agg.errSamples += 1;
        }
        if (candidate.nonFiniteCount > 0) {
          agg.nonFiniteVectors += 1;
        }
        agg.expOverflowEvents += candidate.expOverflowCount;

        const sumErr = Math.abs(sumVector(candidate.values) - 1);
        if (Number.isFinite(sumErr)) {
          agg.probSumAbsErr += sumErr;
          agg.probSumSamples += 1;
        }
      }
    }

    const scaleResult = {};
    for (const [name, agg] of Object.entries(totals)) {
      scaleResult[name] = {
        meanAbsErr: agg.errSamples > 0 ? agg.meanAbsErrSum / agg.errSamples : null,
        nonFiniteVectorRate: agg.nonFiniteVectors / trials,
        meanExpOverflowPerTrial: agg.expOverflowEvents / trials,
        meanProbSumAbsErr: agg.probSumSamples > 0 ? agg.probSumAbsErr / agg.probSumSamples : null,
      };
    }
    byScale[`scale${scale}`] = scaleResult;
  }

  return {
    trials,
    width,
    scales,
    byScale,
  };
}

function main() {
  const dotStudy = runDotStudy();
  const deferredStudy = runDeferredRoundingStudy();
  const overflowStudy = findOverflowPoint();
  const rmsNormStudy = runRmsNormStudy();
  const softmaxStudy = runSoftmaxStudy();

  const result = {
    summary: {
      note: 'Exploratory BF16/F16 numeric study for runtime strategy design.',
      keyTakeaways: [
        'BF16 or F16 inputs with F32 accumulation are far closer to F32 baseline than strict 16-bit accumulation.',
        'Deferring BF16 rounding across multiple updates materially reduces error drift.',
        'F16 overflows many steps earlier than BF16 under multiplicative growth.',
        'RMSNorm is highly sensitive to strict F16 accumulation at larger activation scales.',
        'Softmax without max-subtraction overflows rapidly in strict 16-bit modes.',
      ],
    },
    dotStudy,
    deferredRoundingStudy: deferredStudy,
    overflowStudy,
    rmsNormStudy,
    softmaxStudy,
  };

  console.log(JSON.stringify(result, null, 2));
}

main();
