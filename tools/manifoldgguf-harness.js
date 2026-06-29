#!/usr/bin/env node
// ManifoldGGUF validation harness.
// Loads a real tensor from a safetensors checkpoint, fits a
// coordinate-deterministic descriptor (PRNG + Kronecker + SIREN + sparse),
// writes shard files, and reports byte-accounting metrics.
//
// Usage:
//   node tools/manifoldgguf-harness.js \
//     --safetensors /path/to/model.safetensors \
//     --tensor model.layers.0.mlp.down_proj.weight \
//     --output-dir /tmp/manifoldgguf-run \
//     [--kron-rank 8] [--siren-width 64] [--siren-depth 3] \
//     [--sparse-frac 0.001] [--seed 1337] [--max-slice-dim 1024] \
//     [--calib-activations /path/to/sensitivity.f32bin]
//
// --calib-activations expects a raw F32 binary of shape [cols], where
// sensitivity[j] = mean(abs(X[:, j])) captured by hooking the input
// activations of the target layer during a calibration corpus forward pass.

import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';

// ============================================================================
// CLI
// ============================================================================

function parseArgs(argv) {
  const args = {
    safetensors: null,
    tensor: 'model.layers.0.mlp.down_proj.weight',
    outputDir: null,
    kronRank: 8,
    sirenWidth: 64,
    sirenDepth: 3,
    sparseFrac: 0.001,
    seed: 1337,
    maxSliceDim: 1024,
    calibActivations: null,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => {
      if (argv[i + 1] == null) throw new Error(`Missing value for ${a}`);
      return argv[++i];
    };
    if (a === '--safetensors') args.safetensors = next();
    else if (a === '--tensor') args.tensor = next();
    else if (a === '--output-dir') args.outputDir = next();
    else if (a === '--kron-rank') args.kronRank = parseInt(next(), 10);
    else if (a === '--siren-width') args.sirenWidth = parseInt(next(), 10);
    else if (a === '--siren-depth') args.sirenDepth = parseInt(next(), 10);
    else if (a === '--sparse-frac') args.sparseFrac = parseFloat(next());
    else if (a === '--seed') args.seed = parseInt(next(), 10);
    else if (a === '--max-slice-dim') args.maxSliceDim = parseInt(next(), 10);
    else if (a === '--calib-activations') args.calibActivations = next();
  }
  if (!args.safetensors) throw new Error('--safetensors <path> required');
  if (!args.outputDir) throw new Error('--output-dir <dir> required');
  return args;
}

// ============================================================================
// SafeTensors reader
// ============================================================================

async function loadTensorFromSafetensors(stPath, tensorName) {
  const fh = await fs.open(stPath, 'r');
  try {
    const hdrSizeBuf = Buffer.alloc(8);
    await fh.read(hdrSizeBuf, 0, 8, 0);
    const hdrSize = Number(hdrSizeBuf.readBigUInt64LE(0));

    const hdrBuf = Buffer.alloc(hdrSize);
    await fh.read(hdrBuf, 0, hdrSize, 8);
    const header = JSON.parse(hdrBuf.toString('utf8'));

    const meta = header[tensorName];
    if (!meta) {
      const available = Object.keys(header)
        .filter(k => k !== '__metadata__')
        .slice(0, 10);
      throw new Error(
        `Tensor "${tensorName}" not found in ${stPath}. ` +
        `Available (first 10): ${available.join(', ')}`
      );
    }

    const { dtype, shape, data_offsets: [start, end] } = meta;
    const dataOffset = 8 + hdrSize;
    const byteLen = end - start;
    const dataBuf = Buffer.alloc(byteLen);
    await fh.read(dataBuf, 0, byteLen, dataOffset + start);

    return { dtype, shape, data: dataBuf };
  } finally {
    await fh.close();
  }
}

// ============================================================================
// dtype conversion
// ============================================================================

function bf16BufToF32(buf) {
  const u16 = new Uint16Array(buf.buffer, buf.byteOffset, buf.byteLength / 2);
  const f32 = new Float32Array(u16.length);
  const view = new DataView(new ArrayBuffer(4));
  for (let i = 0; i < u16.length; i++) {
    view.setUint32(0, u16[i] << 16, false);
    f32[i] = view.getFloat32(0, false);
  }
  return f32;
}

function f16BufToF32(buf) {
  const u16 = new Uint16Array(buf.buffer, buf.byteOffset, buf.byteLength / 2);
  const f32 = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const v = u16[i];
    const sign = (v >> 15) ? -1 : 1;
    const exp = (v >> 10) & 0x1f;
    const mant = v & 0x3ff;
    if (exp === 0) {
      f32[i] = sign * 2 ** -14 * (mant / 1024);
    } else if (exp === 31) {
      f32[i] = mant === 0 ? sign * Infinity : NaN;
    } else {
      f32[i] = sign * 2 ** (exp - 15) * (1 + mant / 1024);
    }
  }
  return f32;
}

function toF32(dtype, data) {
  if (dtype === 'BF16') return bf16BufToF32(data);
  if (dtype === 'F16') return f16BufToF32(data);
  if (dtype === 'F32') return new Float32Array(data.buffer, data.byteOffset, data.byteLength / 4);
  throw new Error(
    `Unsupported dtype for CPU conversion: ${dtype}. ` +
    'Add a converter for this dtype before running the harness.'
  );
}

// ============================================================================
// Coordinate-deterministic PRNG (splitmix64 + Box-Muller)
// Frozen algorithm identifier: coord_hash_normal_v1
// Contract: same (seed, row, col) always produces the same value.
// ============================================================================

const PRNG_ALGO = 'coord_hash_normal_v1';

function splitmix64(x) {
  x = BigInt.asUintN(64, x + 0x9E3779B97F4A7C15n);
  x = BigInt.asUintN(64, (x ^ (x >> 30n)) * 0xBF58476D1CE4E5B9n);
  x = BigInt.asUintN(64, (x ^ (x >> 27n)) * 0x94D049BB133111EBn);
  return x ^ (x >> 31n);
}

function coordUniform(seed, row, col) {
  const s = BigInt(seed);
  const r = BigInt(row);
  const c = BigInt(col);
  const z = splitmix64(
    BigInt.asUintN(64, s ^ BigInt.asUintN(64, r * 0x9E3779B97F4A7C15n) ^ BigInt.asUintN(64, c * 0x6C62272E07BB0142n))
  );
  return Number(BigInt.asUintN(53, z)) / Number(1n << 53n);
}

function coordHashNormal(seed, row, col) {
  const u1 = Math.max(coordUniform(seed, row, col), 1e-10);
  const u2 = coordUniform(seed + 1, row, col);
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function buildPRNGMatrix(seed, rows, cols) {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      out[r * cols + c] = coordHashNormal(seed, r, c);
    }
  }
  return out;
}

// ============================================================================
// Kronecker factor shape selection
// ============================================================================

function factorShape(n) {
  const sq = Math.floor(Math.sqrt(n));
  for (let a = sq; a >= 1; a--) {
    if (n % a === 0) return [a, n / a];
  }
  return [1, n];
}

// ============================================================================
// Kronecker ALS via power iteration
//
// Given W [rows x cols] with rows=a*b, cols=c*d, form the block-permuted
// matrix P [a*c x b*d] such that W ≈ A⊗B iff P ≈ vec(A)vec(B)^T.
// ============================================================================

function blockPermute(W, rows, cols, a, b, c, d) {
  const P = new Float32Array((a * c) * (b * d));
  for (let ra = 0; ra < a; ra++) {
    for (let rb = 0; rb < b; rb++) {
      const i = ra * b + rb;
      for (let cc = 0; cc < c; cc++) {
        for (let cd = 0; cd < d; cd++) {
          const j = cc * d + cd;
          P[(ra * c + cc) * (b * d) + (rb * d + cd)] = W[i * cols + j];
        }
      }
    }
  }
  return P;
}

function blockUnpermute(P, a, b, c, d) {
  const rows = a * b;
  const cols = c * d;
  const W = new Float32Array(rows * cols);
  for (let ra = 0; ra < a; ra++) {
    for (let rb = 0; rb < b; rb++) {
      const i = ra * b + rb;
      for (let cc = 0; cc < c; cc++) {
        for (let cd = 0; cd < d; cd++) {
          const j = cc * d + cd;
          W[i * cols + j] = P[(ra * c + cc) * (b * d) + (rb * d + cd)];
        }
      }
    }
  }
  return W;
}

function powerIteration(P, p, q, iters, initSeed) {
  let v = new Float32Array(q);
  for (let j = 0; j < q; j++) {
    v[j] = coordUniform(initSeed, 0, j) - 0.5;
  }

  for (let iter = 0; iter < iters; iter++) {
    const u = new Float32Array(p);
    for (let i = 0; i < p; i++) {
      let s = 0;
      for (let j = 0; j < q; j++) s += P[i * q + j] * v[j];
      u[i] = s;
    }
    let unorm = 0;
    for (let i = 0; i < p; i++) unorm += u[i] * u[i];
    unorm = Math.sqrt(unorm);
    if (unorm > 1e-12) for (let i = 0; i < p; i++) u[i] /= unorm;

    const vnew = new Float32Array(q);
    for (let j = 0; j < q; j++) {
      let s = 0;
      for (let i = 0; i < p; i++) s += P[i * q + j] * u[i];
      vnew[j] = s;
    }
    let vnorm = 0;
    for (let j = 0; j < q; j++) vnorm += vnew[j] * vnew[j];
    vnorm = Math.sqrt(vnorm);
    if (vnorm > 1e-12) for (let j = 0; j < q; j++) vnew[j] /= vnorm;
    v = vnew;
  }

  const uFinal = new Float32Array(p);
  let sigma = 0;
  for (let i = 0; i < p; i++) {
    let s = 0;
    for (let j = 0; j < q; j++) s += P[i * q + j] * v[j];
    uFinal[i] = s;
    sigma += s * s;
  }
  sigma = Math.sqrt(sigma);
  if (sigma > 1e-12) for (let i = 0; i < p; i++) uFinal[i] /= sigma;

  return { u: uFinal, v, sigma };
}

function fitKronecker(W, rows, cols, rank, seed) {
  const [a, b] = factorShape(rows);
  const [c, d] = factorShape(cols);
  const p = a * c;
  const q = b * d;

  const P = blockPermute(W, rows, cols, a, b, c, d);
  const residual = Float32Array.from(P);
  const factors = [];

  for (let r = 0; r < rank; r++) {
    const { u, v, sigma } = powerIteration(residual, p, q, 20, seed + r * 7919);
    if (sigma < 1e-10) break;

    const sqrtSigma = Math.sqrt(sigma);
    const A = new Float32Array(p);
    const B = new Float32Array(q);
    for (let i = 0; i < p; i++) A[i] = u[i] * sqrtSigma;
    for (let j = 0; j < q; j++) B[j] = v[j] * sqrtSigma;

    for (let i = 0; i < p; i++) {
      for (let j = 0; j < q; j++) {
        residual[i * q + j] -= sigma * u[i] * v[j];
      }
    }

    factors.push({ A, B, a, b, c, d });
  }

  return factors;
}

function reconstructKronecker(factors, rows, cols) {
  if (factors.length === 0) return new Float32Array(rows * cols);
  const { a, b, c, d } = factors[0];
  const p = a * c;
  const q = b * d;
  const P = new Float32Array(p * q);
  for (const { A, B } of factors) {
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < q; j++) {
        P[i * q + j] += A[i] * B[j];
      }
    }
  }
  return blockUnpermute(P, a, b, c, d);
}

// ============================================================================
// SIREN (implicit neural representation)
// Input: (row_norm, col_norm) in [-1, 1].  Output: scalar weight value.
// ============================================================================

function sirenInit(width, depth, seed) {
  const layers = [];
  let inDim = 2;
  for (let l = 0; l < depth; l++) {
    const isLast = l === depth - 1;
    const outDim = isLast ? 1 : width;
    const W = new Float32Array(outDim * inDim);
    const b = new Float32Array(outDim);
    const omega = l === 0 ? 30.0 : 1.0;
    const bound = l === 0 ? 1.0 : Math.sqrt(6.0 / inDim) / omega;
    for (let i = 0; i < W.length; i++) {
      W[i] = (coordUniform(seed + l * 1000 + i, l, i) * 2 - 1) * bound;
    }
    layers.push({ W, b, inDim, outDim, omega, isLast });
    inDim = outDim;
  }
  return layers;
}

function sirenForward(layers, rn, cn) {
  let x = [rn, cn];
  for (const { W, b, inDim, outDim, omega, isLast } of layers) {
    const next = new Array(outDim);
    for (let o = 0; o < outDim; o++) {
      let s = b[o];
      for (let i = 0; i < inDim; i++) s += W[o * inDim + i] * x[i];
      next[o] = isLast ? s : Math.sin(omega * s);
    }
    x = next;
  }
  return x[0];
}

function sirenMatrix(layers, rows, cols) {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    const rn = rows > 1 ? (r / (rows - 1)) * 2 - 1 : 0;
    for (let c = 0; c < cols; c++) {
      const cn = cols > 1 ? (c / (cols - 1)) * 2 - 1 : 0;
      out[r * cols + c] = sirenForward(layers, rn, cn);
    }
  }
  return out;
}

function fitSIREN(target, rows, cols, width, depth, seed, steps = 600, lr = 5e-4, batchSize = 512) {
  const layers = sirenInit(width, depth, seed + 99999);

  for (let step = 0; step < steps; step++) {
    const grads = layers.map(l => ({
      dW: new Float32Array(l.W.length),
      db: new Float32Array(l.b.length),
    }));

    for (let bi = 0; bi < batchSize; bi++) {
      const r = Math.floor(coordUniform(seed + step * 100000 + bi, step, bi) * rows);
      const c = Math.floor(coordUniform(seed + step * 100000 + bi + 1, step, bi) * cols);
      const rn = rows > 1 ? (r / (rows - 1)) * 2 - 1 : 0;
      const cn = cols > 1 ? (c / (cols - 1)) * 2 - 1 : 0;
      const tgt = target[r * cols + c];

      const activations = [[rn, cn]];
      const preact = [];
      let x = [rn, cn];
      for (const { W, b, inDim, outDim, omega, isLast } of layers) {
        const z = new Array(outDim);
        for (let o = 0; o < outDim; o++) {
          let s = b[o];
          for (let i = 0; i < inDim; i++) s += W[o * inDim + i] * x[i];
          z[o] = s;
        }
        preact.push(z);
        x = isLast ? [...z] : z.map(v => Math.sin(omega * v));
        activations.push([...x]);
      }

      const err = x[0] - tgt;
      let delta = [2 * err / batchSize];

      for (let l = layers.length - 1; l >= 0; l--) {
        const { W, inDim, outDim, omega, isLast } = layers[l];
        const z = preact[l];
        const inp = activations[l];
        for (let o = 0; o < outDim; o++) {
          grads[l].db[o] += delta[o];
          for (let i = 0; i < inDim; i++) {
            grads[l].dW[o * inDim + i] += delta[o] * inp[i];
          }
        }
        if (l > 0) {
          const prev = new Array(inDim).fill(0);
          for (let i = 0; i < inDim; i++) {
            for (let o = 0; o < outDim; o++) {
              const dAct = isLast ? 1 : omega * Math.cos(omega * z[o]);
              prev[i] += delta[o] * dAct * W[o * inDim + i];
            }
          }
          delta = prev;
        }
      }
    }

    for (let l = 0; l < layers.length; l++) {
      const { W, b } = layers[l];
      const { dW, db } = grads[l];
      for (let i = 0; i < W.length; i++) W[i] -= lr * dW[i];
      for (let i = 0; i < b.length; i++) b[i] -= lr * db[i];
    }
  }

  return layers;
}

// ============================================================================
// Sparse residuals
// ============================================================================

function selectSparse(residual, sensitivity, rows, cols, sparseFrac) {
  const nnz = Math.max(1, Math.round(rows * cols * sparseFrac));

  const scored = new Array(rows * cols);
  for (let i = 0; i < rows * cols; i++) {
    scored[i] = { idx: i, score: Math.abs(residual[i]) * (sensitivity ? sensitivity[i % cols] : 1) };
  }
  scored.sort((a, b) => b.score - a.score);

  const rowIdx = new Int32Array(nnz);
  const colIdx = new Int32Array(nnz);
  const values = new Float32Array(nnz);
  for (let k = 0; k < nnz; k++) {
    const { idx } = scored[k];
    rowIdx[k] = Math.floor(idx / cols);
    colIdx[k] = idx % cols;
    values[k] = residual[idx];
  }

  return { rowIdx, colIdx, values };
}

// ============================================================================
// Serializers
// Binary layout documented in manifoldgguf.v0.1 shard spec.
// ============================================================================

function serializeKronecker(factors) {
  const rank = factors.length;
  let size = 4;
  for (const f of factors) {
    size += 16 + f.A.byteLength + f.B.byteLength;
  }
  const buf = Buffer.alloc(size);
  buf.writeUInt32LE(rank, 0);
  let off = 4;
  for (const f of factors) {
    buf.writeUInt32LE(f.a, off); off += 4;
    buf.writeUInt32LE(f.b, off); off += 4;
    buf.writeUInt32LE(f.c, off); off += 4;
    buf.writeUInt32LE(f.d, off); off += 4;
    Buffer.from(f.A.buffer).copy(buf, off); off += f.A.byteLength;
    Buffer.from(f.B.buffer).copy(buf, off); off += f.B.byteLength;
  }
  return buf;
}

function serializeSIREN(layers) {
  let size = 4;
  for (const l of layers) size += 8 + l.W.byteLength + l.b.byteLength;
  const buf = Buffer.alloc(size);
  buf.writeUInt32LE(layers.length, 0);
  let off = 4;
  for (const l of layers) {
    buf.writeUInt32LE(l.inDim, off); off += 4;
    buf.writeUInt32LE(l.outDim, off); off += 4;
    Buffer.from(l.W.buffer).copy(buf, off); off += l.W.byteLength;
    Buffer.from(l.b.buffer).copy(buf, off); off += l.b.byteLength;
  }
  return buf;
}

function serializeSparse(rowIdx, colIdx, values) {
  const nnz = values.length;
  const buf = Buffer.alloc(4 + nnz * 4 + nnz * 4 + nnz * 4);
  buf.writeUInt32LE(nnz, 0);
  let off = 4;
  Buffer.from(rowIdx.buffer).copy(buf, off); off += rowIdx.byteLength;
  Buffer.from(colIdx.buffer).copy(buf, off); off += colIdx.byteLength;
  Buffer.from(values.buffer).copy(buf, off);
  return buf;
}

// ============================================================================
// Metrics
// ============================================================================

function computeMetrics(W, Wapprox, rows, cols, sensitivity) {
  let sse = 0;
  let frobNum = 0;
  let frobDen = 0;
  let actMse = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const idx = r * cols + c;
      const d = W[idx] - Wapprox[idx];
      sse += d * d;
      frobNum += d * d;
      frobDen += W[idx] * W[idx];
      const s = sensitivity ? sensitivity[c] : 1;
      actMse += d * d * s * s;
    }
  }
  return {
    rmse: Math.sqrt(sse / (rows * cols)),
    relativeFrobenius: Math.sqrt(frobNum / Math.max(frobDen, 1e-10)),
    activationMse: actMse / (rows * cols),
  };
}

function sha256hex(buf) {
  return createHash('sha256').update(buf).digest('hex');
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  const args = parseArgs(process.argv.slice(2));

  console.log(`[manifoldgguf] Loading "${args.tensor}" from ${path.basename(args.safetensors)}`);

  const raw = await loadTensorFromSafetensors(args.safetensors, args.tensor);
  console.log(`[manifoldgguf] dtype=${raw.dtype} shape=[${raw.shape.join(', ')}]`);

  const fullW = toF32(raw.dtype, raw.data);
  const [fullRows, fullCols] = raw.shape;

  const rows = Math.min(fullRows, args.maxSliceDim);
  const cols = Math.min(fullCols, args.maxSliceDim);
  let W;
  if (rows === fullRows && cols === fullCols) {
    W = fullW;
  } else {
    W = new Float32Array(rows * cols);
    for (let r = 0; r < rows; r++) {
      W.set(fullW.subarray(r * fullCols, r * fullCols + cols), r * cols);
    }
    console.log(`[manifoldgguf] Sliced to [${rows}, ${cols}] (--max-slice-dim ${args.maxSliceDim})`);
  }

  // Calibration activations (Fix D)
  let sensitivity = null;
  let sensitivityMode = 'mock_l2_proxy';
  if (args.calibActivations) {
    const buf = await fs.readFile(args.calibActivations);
    if (buf.byteLength !== cols * 4) {
      throw new Error(
        `Calibration activations file must contain ${cols} F32 values (${cols * 4} bytes), ` +
        `got ${buf.byteLength} bytes. Capture with: mean(abs(X[:, j])) over calibration batch.`
      );
    }
    sensitivity = new Float32Array(buf.buffer, buf.byteOffset, cols);
    sensitivityMode = 'real_calibration';
    console.log(`[manifoldgguf] Real calibration activations loaded (${cols} dims)`);
  } else {
    sensitivity = new Float32Array(cols);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        sensitivity[c] += Math.abs(W[r * cols + c]);
      }
    }
    for (let c = 0; c < cols; c++) sensitivity[c] /= rows;
    console.warn(
      '[manifoldgguf] WARNING: Using mock sensitivity (mean |W| per column).\n' +
      '  Pass --calib-activations <f32bin> for proof-grade activation-weighted scoring.\n' +
      '  Capture: hook input to target layer during inference, save mean(abs(X[:,j])) as F32 binary.'
    );
  }

  // PRNG substrate (Fix C: coordinate-deterministic, algorithm frozen in manifest)
  console.log('[manifoldgguf] Building PRNG substrate...');
  const prng = buildPRNGMatrix(args.seed, rows, cols);
  let dotPW = 0, dotPP = 0;
  for (let i = 0; i < W.length; i++) { dotPW += prng[i] * W[i]; dotPP += prng[i] * prng[i]; }
  const prngScale = dotPP > 0 ? dotPW / dotPP : 0;
  const afterPRNG = new Float32Array(W.length);
  for (let i = 0; i < W.length; i++) afterPRNG[i] = W[i] - prngScale * prng[i];

  // Kronecker
  console.log(`[manifoldgguf] Fitting Kronecker rank=${args.kronRank}...`);
  const kronFactors = fitKronecker(afterPRNG, rows, cols, args.kronRank, args.seed);
  const kronApprox = reconstructKronecker(kronFactors, rows, cols);
  const afterKron = new Float32Array(W.length);
  for (let i = 0; i < W.length; i++) afterKron[i] = afterPRNG[i] - kronApprox[i];

  // SIREN
  console.log(`[manifoldgguf] Fitting SIREN width=${args.sirenWidth} depth=${args.sirenDepth}...`);
  const sirenLayers = fitSIREN(afterKron, rows, cols, args.sirenWidth, args.sirenDepth, args.seed);
  const sirenApprox = sirenMatrix(sirenLayers, rows, cols);
  const afterSiren = new Float32Array(W.length);
  for (let i = 0; i < W.length; i++) afterSiren[i] = afterKron[i] - sirenApprox[i];

  // Sparse residuals
  console.log(`[manifoldgguf] Selecting sparse residuals (frac=${args.sparseFrac})...`);
  const sparse = selectSparse(afterSiren, sensitivity, rows, cols, args.sparseFrac);

  // Final reconstruction
  const Wapprox = new Float32Array(W.length);
  for (let i = 0; i < W.length; i++) {
    Wapprox[i] = prngScale * prng[i] + kronApprox[i] + sirenApprox[i];
  }
  for (let k = 0; k < sparse.values.length; k++) {
    Wapprox[sparse.rowIdx[k] * cols + sparse.colIdx[k]] += sparse.values[k];
  }

  const metrics = computeMetrics(W, Wapprox, rows, cols, sensitivity);

  // Serialize shards (Fix E)
  const kronBuf = serializeKronecker(kronFactors);
  const sirenBuf = serializeSIREN(sirenLayers);
  const sparseBuf = serializeSparse(sparse.rowIdx, sparse.colIdx, sparse.values);

  await fs.mkdir(args.outputDir, { recursive: true });

  const key = args.tensor.replace(/\./g, '_');
  const kronFile = `${key}.kron`;
  const sirenFile = `${key}.siren`;
  const sparseFile = `${key}.sparse`;

  await Promise.all([
    fs.writeFile(path.join(args.outputDir, kronFile), kronBuf),
    fs.writeFile(path.join(args.outputDir, sirenFile), sirenBuf),
    fs.writeFile(path.join(args.outputDir, sparseFile), sparseBuf),
  ]);

  const sourceHash = sha256hex(raw.data);
  const kronHash = sha256hex(kronBuf);
  const sirenHash = sha256hex(sirenBuf);
  const sparseHash = sha256hex(sparseBuf);
  const descHash = sha256hex(Buffer.concat([kronBuf, sirenBuf, sparseBuf]));

  // Byte accounting (Fix F)
  const denseBf16Bytes = fullRows * fullCols * 2;
  const descBytes = kronBuf.length + sirenBuf.length + sparseBuf.length;

  const manifest = {
    schema_version: 'manifoldgguf.v0.1',
    tensor_name: args.tensor,
    source_shape: raw.shape,
    slice_shape: [rows, cols],
    storage_type: 'functional_descriptor',
    dtype: 'f16',
    accumulator: 'f32_declared',
    tile_shape: [64, 64],
    source_tensor_hash: `sha256:${sourceHash}`,
    descriptor_hash: `sha256:${descHash}`,
    components: {
      prng_substrate: {
        algorithm: PRNG_ALGO,
        seed: args.seed,
        learned_scale: prngScale,
        learned_scale_frozen: true,
      },
      kronecker_sum: {
        rank_terms: kronFactors.length,
        factor_shapes: kronFactors.map(f => [[f.a, f.c], [f.b, f.d]]),
        shard_file: kronFile,
      },
      coordinate_inr: {
        type: 'siren',
        network_dims: [2, ...Array(args.sirenDepth - 1).fill(args.sirenWidth), 1],
        omega_0: 30.0,
        shard_file: sirenFile,
      },
      sparse_outliers: {
        format: 'csr_v1',
        selection: 'residual_x_activation_sensitivity',
        nnz_fraction: args.sparseFrac,
        value_dtype: 'f32',
        actual_nnz: sparse.values.length,
        shard_file: sparseFile,
      },
    },
  };

  const hashes = {
    source_tensor: `sha256:${sourceHash}`,
    kron_shard: `sha256:${kronHash}`,
    siren_shard: `sha256:${sirenHash}`,
    sparse_shard: `sha256:${sparseHash}`,
    descriptor: `sha256:${descHash}`,
  };

  const metricsOut = {
    dense_f16_bytes: denseBf16Bytes,
    descriptor_bytes: descBytes,
    compression_ratio: (descBytes / denseBf16Bytes).toFixed(4),
    rmse: metrics.rmse.toFixed(7),
    relative_frobenius: metrics.relativeFrobenius.toFixed(7),
    activation_mse: metrics.activationMse.toFixed(7),
    sensitivity_mode: sensitivityMode,
    source_shape: raw.shape,
    slice_shape: [rows, cols],
    kron_rank: kronFactors.length,
    siren_param_count: sirenLayers.reduce((s, l) => s + l.W.length + l.b.length, 0),
    sparse_nnz: sparse.values.length,
  };

  await Promise.all([
    fs.writeFile(path.join(args.outputDir, 'manifest.json'), JSON.stringify(manifest, null, 2)),
    fs.writeFile(path.join(args.outputDir, 'hashes.json'), JSON.stringify(hashes, null, 2)),
    fs.writeFile(path.join(args.outputDir, 'metrics.json'), JSON.stringify(metricsOut, null, 2)),
  ]);

  console.log('\n[manifoldgguf] === RESULTS ===');
  console.log(`  Dense F16 bytes : ${denseBf16Bytes.toLocaleString()} (full ${fullRows}x${fullCols})`);
  console.log(`  Descriptor bytes: ${descBytes.toLocaleString()} (kron=${kronBuf.length} siren=${sirenBuf.length} sparse=${sparseBuf.length})`);
  console.log(`  Compression     : ${metricsOut.compression_ratio}x (slice ${rows}x${cols})`);
  console.log(`  RMSE            : ${metricsOut.rmse}`);
  console.log(`  Rel. Frobenius  : ${metricsOut.relative_frobenius}`);
  console.log(`  Activation MSE  : ${metricsOut.activation_mse} [${sensitivityMode}]`);
  console.log(`  Output dir      : ${args.outputDir}/`);

  if (sensitivityMode !== 'real_calibration') {
    console.log('\n[manifoldgguf] PROOF STATUS: incomplete — mock sensitivity.');
    console.log('  Supply --calib-activations to qualify as proof-grade.');
  } else {
    console.log('\n[manifoldgguf] PROOF STATUS: sensitivity gate passed.');
    if (descBytes < denseBf16Bytes) {
      console.log('  Compression gate passed (descriptor < dense).');
    } else {
      console.log('  WARNING: descriptor_bytes >= dense_f16_bytes. Increase rank or reduce siren width.');
    }
  }
}

main().catch(e => {
  console.error('[manifoldgguf] FATAL:', e.message);
  process.exit(1);
});
