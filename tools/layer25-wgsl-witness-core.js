// Offline diagnostic witness for Layer-25 WGSL operators. No runtime imports or qualification authority.
import { dyadic, add, multiply } from './dyadic-diagnostic.js';

const f32Buf = new ArrayBuffer(4);
const f32View = new Float32Array(f32Buf);
const u32View = new Uint32Array(f32Buf);
const i32View = new Int32Array(f32Buf);

const f64Buf = new ArrayBuffer(8);
const f64View = new Float64Array(f64Buf);
const f64Words = new Uint32Array(f64Buf);

const r = Math.fround;

export function f32ToBits(x) {
  f32View[0] = x;
  return u32View[0] >>> 0;
}

export function bitsToF32(u) {
  u32View[0] = u >>> 0;
  return f32View[0];
}

function bitsToI32(u) {
  u32View[0] = u >>> 0;
  return i32View[0];
}

function firstLeadingBitU32(u) {
  const word = u >>> 0;
  return word === 0 ? 0xffffffff : (31 - Math.clz32(word));
}

// WGSL round(x) is roundTiesToEven, unlike JS Math.round(x).
export function roundTiesToEven(x) {
  if (!Number.isFinite(x)) throw new Error('Finite input required for roundTiesToEven');
  const floor = Math.floor(x);
  const diff = x - floor;
  if (diff < 0.5) return floor;
  if (diff > 0.5) return floor + 1;
  return (floor % 2 === 0) ? floor : floor + 1;
}

// Single-rounded IEEE-754 binary32 fused multiply-add:
// For binary32 inputs a and b, a * b has at most 48 significand bits and is exact in binary64.
// Double rounding from binary64 to binary32 can only occur if the binary64 sum s = p + c lands on
// an exact binary32 midpoint (low 29 mantissa bits == 0x10000000) while the TwoSum residual err != 0.
export function fmaExact(a, b, c) {
  const ax = r(a);
  const bx = r(b);
  const cx = r(c);
  const p = ax * bx;
  const s = p + cx;
  f64View[0] = s;
  const lo = f64Words[0] >>> 0;
  if ((lo & 0x1fffffff) !== 0x10000000) {
    return r(s);
  }
  const t = s - p;
  const err = (p - (s - t)) + (cx - t);
  if (err === 0) {
    return r(s);
  }
  if ((s > 0 && err > 0) || (s < 0 && err < 0)) {
    f64Words[0] = (lo | 1) >>> 0;
  } else {
    f64Words[0] = (lo - 1) >>> 0;
  }
  return r(f64View[0]);
}

export function reciprocalRefined(x) {
  const xv = r(x);
  const est = r(1.0 / xv);
  return fmaExact(est, fmaExact(-xv, est, 1.0), est);
}

export function divideRefined(x, y) {
  const xv = r(x);
  const yv = r(y);
  const inv = reciprocalRefined(yv);
  const q = r(xv * inv);
  return fmaExact(fmaExact(-q, yv, xv), inv, q);
}

export function inverseRootRefined(x, estimateOverride = null) {
  const xv = r(x);
  const est = estimateOverride === null ? r(1.0 / Math.sqrt(xv)) : r(estimateOverride);
  const sq = r(est * est);
  const err = fmaExact(est, est, -sq);
  const res = r(fmaExact(-xv, sq, 1.0) - r(xv * err));
  return fmaExact(r(0.5 * est), res, est);
}

export function expRefined(x) {
  const xv = r(x);
  const ax = Math.abs(xv);
  if (ax > 80.0) return r(Math.exp(xv));
  const n = roundTiesToEven(r(xv * r(1.4426950408889634)));
  let rem = xv;
  if (ax >= 0.000000059604644775390625) {
    const bits = f32ToBits(ax);
    const mant = ((bits & 0x007fffff) | 0x00800000) >>> 0;
    const shift = ((bits >>> 23) & 255) - 118;
    let inFixed = 0;
    if (shift >= 0) {
      inFixed = (mant << shift) >>> 0;
    } else {
      const right = -shift;
      inFixed = ((mant + (1 << (right - 1))) >>> right) >>> 0;
    }
    const powFixed = Math.imul(Math.abs(n) >>> 0, 2977044472) >>> 0;
    const res = (xv >= 0.0 ? (inFixed - powFixed) : (powFixed - inFixed)) >>> 0;
    rem = r(r(bitsToI32(res)) * r(0.00000000023283064365386962890625));
  }
  let p = r(0.0001984126984126984);
  p = fmaExact(p, rem, r(0.001388888888888889));
  p = fmaExact(p, rem, r(0.008333333333333333));
  p = fmaExact(p, rem, r(0.041666666666666664));
  p = fmaExact(p, rem, r(0.16666666666666666));
  p = fmaExact(p, rem, 0.5);
  p = fmaExact(p, rem, 1.0);
  p = fmaExact(p, rem, 1.0);
  return r(p * bitsToF32(((n + 127) << 23) >>> 0));
}

export function siluWgsl(x) {
  const xv = r(x);
  const z = expRefined(-Math.abs(xv));
  const num = xv >= 0.0 ? xv : r(xv * z);
  return r(num * reciprocalRefined(r(1.0 + z)));
}

export function sigmoidRefined(x) {
  const xv = r(x);
  const z = expRefined(-Math.abs(xv));
  const rec = reciprocalRefined(r(1.0 + z));
  return xv >= 0.0 ? rec : r(z * rec);
}

export function logRefined(x) {
  const xv = r(x);
  if (xv <= 0.0 || xv > 3.4028234663852886e38) return r(Math.log(xv));
  const bits = f32ToBits(xv);
  let exp = ((bits >>> 23) & 255) - 127;
  let mant = bitsToF32(((bits & 0x007fffff) | 0x3f800000) >>> 0);
  if (mant > r(1.4142135623730951)) {
    mant = r(mant * 0.5);
    exp += 1;
  }
  const s = r(r(mant - 1.0) * reciprocalRefined(r(mant + 1.0)));
  const sq = r(s * s);
  let p = r(0.09090909090909091);
  p = fmaExact(p, sq, r(0.1111111111111111));
  p = fmaExact(p, sq, r(0.14285714285714285));
  p = fmaExact(p, sq, r(0.2));
  p = fmaExact(p, sq, r(0.3333333333333333));
  p = fmaExact(p, sq, 1.0);
  const reduced = r(r(2.0 * s) * p);
  if (exp === 0) return reduced;
  const n = Math.abs(exp) >>> 0;
  const bottom = Math.imul(n, 2977044472 & 65535) >>> 0;
  const top = Math.imul(n, 2977044472 >>> 16) >>> 0;
  let lo = (bottom + ((top << 16) >>> 0)) >>> 0;
  let hi = ((top >>> 16) + (lo < bottom ? 1 : 0)) >>> 0;
  if (exp < 0) {
    lo = (0 - lo) >>> 0;
    hi = ((~hi) + (lo === 0 ? 1 : 0)) >>> 0;
  }
  const rem = roundTiesToEven(r(reduced * 4294967296.0)) | 0;
  const oldLo = lo;
  lo = (lo + (rem >>> 0)) >>> 0;
  hi = (hi + (rem >= 0 ? 0 : 0xffffffff) + (lo < oldLo ? 1 : 0)) >>> 0;
  const neg = (hi & 0x80000000) !== 0;
  if (neg) {
    lo = (0 - lo) >>> 0;
    hi = ((~hi) + (lo === 0 ? 1 : 0)) >>> 0;
  }
  const leading = hi !== 0 ? (32 + firstLeadingBitU32(hi)) : firstLeadingBitU32(lo);
  const shift = leading - 23;
  let m = ((lo >>> shift) | (shift === 0 ? 0 : (hi << (32 - shift)))) >>> 0;
  const disc = (lo & ((1 << shift) - 1)) >>> 0;
  const half = (1 << (shift - 1)) >>> 0;
  if (disc > half || (disc === half && (m & 1) !== 0)) {
    m = (m + 1) >>> 0;
  }
  const mag = r(r(m) * bitsToF32(((shift + 95) << 23) >>> 0));
  return neg ? -mag : mag;
}

export function softplusWgsl(x) {
  const xv = r(x);
  if (xv > 20.0) return xv;
  if (xv < -20.0) return expRefined(xv);
  return logRefined(r(1.0 + expRefined(xv)));
}

export function witnessHyperConnectionNorm(input, weight, rows, options = {}) {
  const streamCount = options.streamCount ?? 4;
  const streamWidth = options.streamWidth ?? 2560;
  const eps = r(options.eps ?? 1e-6);
  const weightOffset = options.weightOffset ?? true;
  const rightAssociated = options.rightAssociated ?? false;
  const totalGroups = rows * streamCount;
  if (input.length !== totalGroups * streamWidth || weight.length !== streamCount * streamWidth) {
    throw new Error('hyper_connection_norm shape mismatch');
  }
  const out = new Float32Array(input.length);
  const partial = new Float32Array(256);
  for (let group = 0; group < totalGroups; group++) {
    const base = group * streamWidth;
    const wBase = (group % streamCount) * streamWidth;
    for (let local = 0; local < 256; local++) {
      let sum = 0.0;
      for (let col = local; col < streamWidth; col += 256) {
        const v = input[base + col];
        sum = fmaExact(v, v, sum);
      }
      partial[local] = sum;
    }
    for (let stride = 128; stride > 0; stride >>= 1) {
      for (let local = 0; local < stride; local++) {
        partial[local] = r(partial[local] + partial[local + stride]);
      }
    }
    const meanSq = divideRefined(partial[0], streamWidth);
    const invRms = inverseRootRefined(r(meanSq + eps));
    for (let col = 0; col < streamWidth; col++) {
      const x = input[base + col];
      const wScale = weightOffset ? r(1.0 + weight[wBase + col]) : weight[wBase + col];
      out[base + col] = rightAssociated
        ? r(x * r(invRms * wScale))
        : r(r(x * invRms) * wScale);
    }
  }
  return out;
}

export function witnessLowRankSilu(input, streamCount = 4) {
  const out = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) {
    out[i] = siluWgsl(divideRefined(input[i], streamCount));
  }
  return out;
}

export function witnessMixSigmoid(input) {
  const out = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) {
    out[i] = sigmoidRefined(input[i]);
  }
  return out;
}

export function witnessInjectGate(input, streamCount = 4) {
  const out = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) {
    out[i] = r(2.0 * sigmoidRefined(divideRefined(input[i], streamCount)));
  }
  return out;
}

export function witnessHyperConnectionBranch(normed, mix, rows, streamCount = 4, streamWidth = 2560) {
  const rowStride = streamCount * streamWidth;
  if (normed.length !== rows * rowStride || mix.length !== rows * rowStride) {
    throw new Error('hyper_connection_branch shape mismatch');
  }
  const out = new Float32Array(rows * streamWidth);
  for (let row = 0; row < rows; row++) {
    const rBase = row * rowStride;
    const oBase = row * streamWidth;
    for (let col = 0; col < streamWidth; col++) {
      let sum = 0.0;
      for (let s = 0; s < streamCount; s++) {
        const idx = rBase + s * streamWidth + col;
        const prod = r(normed[idx] * mix[idx]);
        sum = r(sum + prod);
      }
      out[oBase + col] = divideRefined(sum, streamCount);
    }
  }
  return out;
}

export function witnessHyperConnectionInject(streams, branch, gate, rows, streamCount = 4, streamWidth = 2560) {
  const rowStride = streamCount * streamWidth;
  if (streams.length !== rows * rowStride || branch.length !== rows * streamWidth || gate.length !== rows * streamCount) {
    throw new Error('hyper_connection_inject shape mismatch');
  }
  const out = new Float32Array(rows * rowStride);
  for (let row = 0; row < rows; row++) {
    const rBase = row * rowStride;
    const bBase = row * streamWidth;
    const gBase = row * streamCount;
    for (let s = 0; s < streamCount; s++) {
      const g = gate[gBase + s];
      const sBase = rBase + s * streamWidth;
      for (let col = 0; col < streamWidth; col++) {
        const scaled = r(branch[bBase + col] * g);
        out[sBase + col] = r(streams[sBase + col] + scaled);
      }
    }
  }
  return out;
}

// matmul_f32_row_subgroup.wgsl on wave64 (AMD RDNA 3.5):
// Each lane accumulates vec4<f32> with FMA at stride 64 * 4 = 256.
// RADV/ACO lowers subgroupAdd(vec4<f32>) to a 6-stage wave64 DPP binary tree (stride = 1, 2, 4, 8, 16, 32).
export function witnessMatmulRowSubgroupWave64(input, rows, K, weight, N, options = {}) {
  if (input.length !== rows * K || weight.length !== N * K) {
    throw new Error('matmul_f32_row_subgroup shape mismatch');
  }
  const topDown = options.reductionOrder === 'top-down';
  const out = new Float32Array(rows * N);
  const lx = new Float32Array(64);
  const ly = new Float32Array(64);
  const lz = new Float32Array(64);
  const lw = new Float32Array(64);
  for (let row = 0; row < rows; row++) {
    const xBase = row * K;
    for (let col = 0; col < N; col++) {
      const wBase = col * K;
      for (let lane = 0; lane < 64; lane++) {
        let ax = 0.0;
        let ay = 0.0;
        let az = 0.0;
        let aw = 0.0;
        for (let k = lane * 4; k < K; k += 256) {
          ax = fmaExact(input[xBase + k], weight[wBase + k], ax);
          ay = fmaExact(input[xBase + k + 1], weight[wBase + k + 1], ay);
          az = fmaExact(input[xBase + k + 2], weight[wBase + k + 2], az);
          aw = fmaExact(input[xBase + k + 3], weight[wBase + k + 3], aw);
        }
        lx[lane] = ax;
        ly[lane] = ay;
        lz[lane] = az;
        lw[lane] = aw;
      }
      if (topDown) {
        for (let s = 32; s >= 1; s >>= 1) {
          for (let i = 0; i < s; i++) {
            lx[i] = r(lx[i] + lx[i + s]);
            ly[i] = r(ly[i] + ly[i + s]);
            lz[i] = r(lz[i] + lz[i + s]);
            lw[i] = r(lw[i] + lw[i + s]);
          }
        }
      } else {
        for (let s = 1; s < 64; s <<= 1) {
          for (let i = 0; i < 64 - s; i++) {
            lx[i] = r(lx[i] + lx[i + s]);
            ly[i] = r(ly[i] + ly[i + s]);
            lz[i] = r(lz[i] + lz[i + s]);
            lw[i] = r(lw[i] + lw[i + s]);
          }
        }
      }
      out[row * N + col] = r(r(r(lx[0] + ly[0]) + lz[0]) + lw[0]);
    }
  }
  return out;
}

// matmul_f16w_f32a_tiled.wgsl:
// Bound each local accumulation to one K-tile of 16 elements via FMA into tile = 0.0, then acc += tile.
export function witnessMatmulF16wF32aTiled16(input, rows, K, weight, N) {
  if (input.length !== rows * K || weight.length !== N * K) {
    throw new Error('matmul_f16w_f32a_tiled shape mismatch');
  }
  const out = new Float32Array(rows * N);
  for (let row = 0; row < rows; row++) {
    const xBase = row * K;
    for (let col = 0; col < N; col++) {
      const wBase = col * K;
      let acc = 0.0;
      for (let t = 0; t < K; t += 16) {
        let tile = 0.0;
        const limit = Math.min(16, K - t);
        for (let k = 0; k < limit; k++) {
          tile = fmaExact(input[xBase + t + k], weight[wBase + t + k], tile);
        }
        acc = r(acc + tile);
      }
      out[row * N + col] = acc;
    }
  }
  return out;
}

// matmul_f16w_f32a.wgsl (selected when N < 64, e.g. in_proj_a and in_proj_b with N = 48):
// Sequential accumulation over 0..K-1.
export function witnessMatmulF16wF32aSeq(input, rows, K, weight, N, options = {}) {
  if (input.length !== rows * K || weight.length !== N * K) {
    throw new Error('matmul_f16w_f32a shape mismatch');
  }
  const useFma = options.useFma ?? true;
  const out = new Float32Array(rows * N);
  for (let row = 0; row < rows; row++) {
    const xBase = row * K;
    for (let col = 0; col < N; col++) {
      const wBase = col * K;
      let acc = 0.0;
      for (let k = 0; k < K; k++) {
        acc = useFma
          ? fmaExact(input[xBase + k], weight[wBase + k], acc)
          : r(acc + r(input[xBase + k] * weight[wBase + k]));
      }
      out[row * N + col] = acc;
    }
  }
  return out;
}

// gated_delta_conv.wgsl + gated_delta_recurrent_sigmoid_gate.wgsl
export function witnessGatedDeltaCore(
  qkv,
  zProj,
  aProj,
  bProj,
  convWeight,
  dtBias,
  aLog,
  normWeight,
  numTokens,
  options = {}
) {
  const convDim = options.convDim ?? 10240;
  const kernelSize = options.kernelSize ?? 4;
  const numVHeads = options.numVHeads ?? 48;
  const headKDim = options.headKDim ?? 128;
  const headVDim = options.headVDim ?? 128;
  const qSize = options.qSize ?? 2048;
  const kSize = options.kSize ?? 2048;
  const valueDim = options.valueDim ?? 6144;
  const qRep = options.qRep ?? 3;
  const rmsEps = r(options.rmsEps ?? 1e-6);
  const qkEps = r(options.qkEps ?? 1e-6);
  const convFma = options.convFma ?? true;
  const gateActivation = options.gateActivation ?? 'sigmoid';

  if (qkv.length !== numTokens * convDim || zProj.length !== numTokens * valueDim ||
      aProj.length !== numTokens * numVHeads || bProj.length !== numTokens * numVHeads) {
    throw new Error('gated_delta_core input shape mismatch');
  }

  const convOut = new Float32Array(numTokens * convDim);
  const stateRing = new Float32Array(kernelSize);
  for (let ch = 0; ch < convDim; ch++) {
    const sb = ch * kernelSize;
    stateRing.fill(0.0);
    for (let t = 0; t < numTokens; t++) {
      for (let k = 0; k + 1 < kernelSize; k++) {
        stateRing[k] = stateRing[k + 1];
      }
      stateRing[kernelSize - 1] = qkv[t * convDim + ch];
      let mixed = 0.0;
      for (let k = 0; k < kernelSize; k++) {
        mixed = convFma
          ? fmaExact(stateRing[k], convWeight[sb + k], mixed)
          : r(mixed + r(stateRing[k] * convWeight[sb + k]));
      }
      convOut[t * convDim + ch] = siluWgsl(mixed);
    }
  }

  const output = new Float32Array(numTokens * valueDim);
  const recState = new Float32Array(numVHeads * headKDim * headVDim);
  const headScale = r(1.0 / Math.sqrt(headKDim));
  const sharedSq = new Float32Array(headVDim);
  const kNormed = new Float32Array(headKDim);
  const qNormed = new Float32Array(headKDim);

  for (let head = 0; head < numVHeads; head++) {
    const recBase = head * headKDim * headVDim;
    const srcHead = Math.trunc(head / qRep);
    const qBase = srcHead * headKDim;
    const kBase = qSize + srcHead * headKDim;
    const vBase = qSize + kSize + head * headVDim;
    const expALog = expRefined(aLog[head]);
    const biasH = dtBias[head];

    for (let t = 0; t < numTokens; t++) {
      const convRow = t * convDim;
      const zRow = t * valueDim + head * headVDim;
      const abIdx = t * numVHeads + head;
      const outRow = t * valueDim + head * headVDim;

      for (let vd = 0; vd < headVDim; vd++) {
        const qv = convOut[convRow + qBase + vd];
        sharedSq[vd] = r(qv * qv);
      }
      for (let s = headVDim >> 1; s > 0; s >>= 1) {
        for (let vd = 0; vd < s; vd++) {
          sharedSq[vd] = r(sharedSq[vd] + sharedSq[vd + s]);
        }
      }
      const qScale = r(headScale * inverseRootRefined(r(sharedSq[0] + qkEps)));

      for (let vd = 0; vd < headVDim; vd++) {
        const kv = convOut[convRow + kBase + vd];
        sharedSq[vd] = r(kv * kv);
      }
      for (let s = headVDim >> 1; s > 0; s >>= 1) {
        for (let vd = 0; vd < s; vd++) {
          sharedSq[vd] = r(sharedSq[vd] + sharedSq[vd + s]);
        }
      }
      const kScale = inverseRootRefined(r(sharedSq[0] + qkEps));

      const beta = sigmoidRefined(bProj[abIdx]);
      const g = r(-expALog * softplusWgsl(r(aProj[abIdx] + biasH)));
      const gExp = expRefined(g);

      for (let kd = 0; kd < headKDim; kd++) {
        kNormed[kd] = r(convOut[convRow + kBase + kd] * kScale);
        qNormed[kd] = r(convOut[convRow + qBase + kd] * qScale);
      }

      for (let vd = 0; vd < headVDim; vd++) {
        let kvMem = 0.0;
        for (let kd = 0; kd < headKDim; kd++) {
          const sidx = recBase + kd * headVDim + vd;
          const decayed = r(recState[sidx] * gExp);
          recState[sidx] = decayed;
          kvMem = fmaExact(decayed, kNormed[kd], kvMem);
        }
        const delta = r(r(convOut[convRow + vBase + vd] - kvMem) * beta);
        let outVal = 0.0;
        for (let kd = 0; kd < headKDim; kd++) {
          const sidx = recBase + kd * headVDim + vd;
          const updated = fmaExact(kNormed[kd], delta, recState[sidx]);
          recState[sidx] = updated;
          outVal = fmaExact(updated, qNormed[kd], outVal);
        }
        output[outRow + vd] = outVal;
        sharedSq[vd] = r(outVal * outVal);
      }

      for (let s = headVDim >> 1; s > 0; s >>= 1) {
        for (let vd = 0; vd < s; vd++) {
          sharedSq[vd] = r(sharedSq[vd] + sharedSq[vd + s]);
        }
      }
      const invRms = inverseRootRefined(r(r(sharedSq[0] / headVDim) + rmsEps));

      for (let vd = 0; vd < headVDim; vd++) {
        const zVal = zProj[zRow + vd];
        const gate = gateActivation === 'sigmoid' ? sigmoidRefined(zVal) : siluWgsl(zVal);
        let val = r(output[outRow + vd] * invRms);
        val = r(val * normWeight[vd]);
        val = r(val * gate);
        output[outRow + vd] = val;
      }
    }
  }

  return output;
}

export function compareF32Bits(predicted, actual) {
  if (predicted.length !== actual.length || !predicted.length) {
    throw new Error('compareF32Bits length mismatch');
  }
  let exact = 0;
  let maxUlp = 0;
  let maxAbsDiff = 0;
  for (let i = 0; i < predicted.length; i++) {
    const pv = predicted[i];
    const av = actual[i];
    if (!Number.isFinite(pv) || !Number.isFinite(av)) {
      throw new Error('Nonfinite value encountered in compareF32Bits');
    }
    const ub = f32ToBits(av);
    const pb = f32ToBits(pv);
    if (pb === ub) {
      exact++;
    } else {
      const ulp = Math.abs(pb - ub);
      if (ulp > maxUlp) maxUlp = ulp;
      const diff = Math.abs(pv - av);
      if (diff > maxAbsDiff) maxAbsDiff = diff;
    }
  }
  return {
    exact,
    total: predicted.length,
    allExact: exact === predicted.length,
    maxUlp,
    maxAbsDiff,
  };
}

export function exactDyadicDot(inputRow, weightRow) {
  if (inputRow.length !== weightRow.length || !inputRow.length) {
    throw new Error('exactDyadicDot length mismatch');
  }
  let sum = { n: 0n, e: 0 };
  for (let i = 0; i < inputRow.length; i++) {
    sum = add(sum, multiply(dyadic(inputRow[i]), dyadic(weightRow[i])));
  }
  return Number(sum.n) * (2 ** sum.e);
}

export function exactDyadicPairMargin(inputRow, weightRowA, weightRowB) {
  if (inputRow.length !== weightRowA.length || inputRow.length !== weightRowB.length) {
    throw new Error('exactDyadicPairMargin length mismatch');
  }
  const dotA = exactDyadicDot(inputRow, weightRowA);
  const dotB = exactDyadicDot(inputRow, weightRowB);
  let diffSum = { n: 0n, e: 0 };
  for (let i = 0; i < inputRow.length; i++) {
    const x = dyadic(inputRow[i]);
    const wDiff = add(dyadic(weightRowA[i]), { n: -dyadic(weightRowB[i]).n, e: dyadic(weightRowB[i]).e });
    diffSum = add(diffSum, multiply(x, wDiff));
  }
  return {
    dotA,
    dotB,
    margin: Number(diffSum.n) * (2 ** diffSum.e),
  };
}
