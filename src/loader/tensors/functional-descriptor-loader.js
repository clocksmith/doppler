import { acquireBuffer, uploadData } from '../../memory/buffer-pool.js';
import { createWeightBuffer } from '../../gpu/weight-buffer.js';
import { assertFunctionalDescriptorManifest } from '../../formats/rdrr/functional-descriptor.js';
import { log } from '../../debug/index.js';

// Reconstruct a functional_descriptor tensor from its component shards and
// upload the result to a GPU weight buffer. All reconstruction runs on CPU;
// the result is a row-major F32 (or F16 when hardware supports it) buffer.
//
// shardData must carry descriptorShards: Map<string, Uint8Array> keyed by the
// shard_file names declared in the descriptor manifest (kron, siren, sparse).
// location must carry descriptorManifest: the parsed manifoldgguf.v0.1 object.

// ============================================================================
// Coord-deterministic PRNG — must match tools/manifoldgguf-harness.js exactly.
// Algorithm identifier: coord_hash_normal_v1
// ============================================================================

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

// ============================================================================
// PRNG component
// ============================================================================

function buildPRNG(prngSpec, rows, cols) {
  if (prngSpec.algorithm !== 'coord_hash_normal_v1') {
    throw new Error(
      `[FunctionalDescriptor] Unknown PRNG algorithm "${prngSpec.algorithm}". ` +
      'Only coord_hash_normal_v1 is supported in this runtime version.'
    );
  }
  const { seed, learned_scale: scale } = prngSpec;
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      out[r * cols + c] = scale * coordHashNormal(seed, r, c);
    }
  }
  return out;
}

// ============================================================================
// Kronecker component
// Binary layout: [rank u32][a,b,c,d u32x4][A f32...][B f32...] per term
// ============================================================================

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

function deserializeKronecker(buf) {
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const rank = view.getUint32(0, true);
  let off = 4;
  const factors = [];
  for (let r = 0; r < rank; r++) {
    const a = view.getUint32(off, true); off += 4;
    const b = view.getUint32(off, true); off += 4;
    const c = view.getUint32(off, true); off += 4;
    const d = view.getUint32(off, true); off += 4;
    const aLen = a * c;
    const bLen = b * d;
    const A = new Float32Array(buf.buffer, buf.byteOffset + off, aLen); off += aLen * 4;
    const B = new Float32Array(buf.buffer, buf.byteOffset + off, bLen); off += bLen * 4;
    factors.push({ a, b, c, d, A, B });
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
// SIREN component
// Binary layout: [depth u32][inDim,outDim u32x2][W f32...][b f32...] per layer
// ============================================================================

function deserializeSIREN(buf) {
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const depth = view.getUint32(0, true);
  let off = 4;
  const layers = [];
  for (let l = 0; l < depth; l++) {
    const inDim = view.getUint32(off, true); off += 4;
    const outDim = view.getUint32(off, true); off += 4;
    const W = new Float32Array(buf.buffer, buf.byteOffset + off, inDim * outDim); off += inDim * outDim * 4;
    const b = new Float32Array(buf.buffer, buf.byteOffset + off, outDim); off += outDim * 4;
    const isLast = l === depth - 1;
    layers.push({ inDim, outDim, W, b, omega: l === 0 ? 30.0 : 1.0, isLast });
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

function reconstructSIREN(layers, rows, cols) {
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

function cropMatrix(matrix, rows, cols, cropRows, cropCols) {
  if (rows === cropRows && cols === cropCols) {
    return matrix;
  }
  const out = new Float32Array(cropRows * cropCols);
  for (let r = 0; r < cropRows; r += 1) {
    out.set(
      matrix.subarray(r * cols, (r * cols) + cropCols),
      r * cropCols
    );
  }
  return out;
}

function normalizeShape2(value, label) {
  if (!Array.isArray(value) || value.length !== 2) {
    throw new Error(`[FunctionalDescriptor] ${label} must be [rows, cols].`);
  }
  const rows = Number(value[0]);
  const cols = Number(value[1]);
  if (!Number.isInteger(rows) || rows <= 0 || !Number.isInteger(cols) || cols <= 0) {
    throw new Error(`[FunctionalDescriptor] ${label} has invalid shape [${value.join(',')}].`);
  }
  return [rows, cols];
}

function getDescriptorShardBytes(shards) {
  let total = 0;
  for (const bytes of shards.values()) {
    total += bytes.byteLength;
  }
  return total;
}

function assertOptionalIntegerEquals(value, expected, label, name) {
  if (value === undefined) {
    return;
  }
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed !== expected) {
    throw new Error(
      `[FunctionalDescriptor] ${label} for "${name}" must equal ${expected}, got ${value}.`
    );
  }
}

function assertRuntimeProofGates(manifest, name, descriptorBytes, denseF16Bytes) {
  assertOptionalIntegerEquals(manifest.descriptor_bytes, descriptorBytes, 'descriptor_bytes', name);
  assertOptionalIntegerEquals(manifest.dense_f16_bytes, denseF16Bytes, 'dense_f16_bytes', name);

  const gate = manifest.proof_status_gate;
  const proofStatus = typeof manifest.proof_status === 'string'
    ? manifest.proof_status.trim().toLowerCase()
    : null;
  const sensitivityGate = typeof gate?.sensitivity === 'string'
    ? gate.sensitivity.trim().toLowerCase()
    : null;
  const compressionGate = typeof gate?.compression === 'string'
    ? gate.compression.trim().toLowerCase()
    : null;

  if (proofStatus === 'passed' && sensitivityGate !== 'passed') {
    throw new Error(
      `[FunctionalDescriptor] proof_status is "passed" for "${name}" but proof_status_gate.sensitivity is not "passed".`
    );
  }
  if (proofStatus === 'passed' && compressionGate !== 'passed') {
    throw new Error(
      `[FunctionalDescriptor] proof_status is "passed" for "${name}" but proof_status_gate.compression is not "passed".`
    );
  }
  if (compressionGate === 'passed' && descriptorBytes >= denseF16Bytes) {
    throw new Error(
      `[FunctionalDescriptor] compression proof gate failed for "${name}": ` +
      `descriptor_bytes=${descriptorBytes} must be lower than dense_f16_bytes=${denseF16Bytes}.`
    );
  }
}

// ============================================================================
// Sparse component (coo_v1)
// Binary layout: [nnz u32][rowIdx i32...][colIdx i32...][values f32...]
// ============================================================================

function deserializeSparse(buf) {
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const nnz = view.getUint32(0, true);
  let off = 4;
  const rowIdx = new Int32Array(buf.buffer, buf.byteOffset + off, nnz); off += nnz * 4;
  const colIdx = new Int32Array(buf.buffer, buf.byteOffset + off, nnz); off += nnz * 4;
  const values = new Float32Array(buf.buffer, buf.byteOffset + off, nnz);
  return { rowIdx, colIdx, values };
}

function applySparse(out, sparse, cols) {
  const { rowIdx, colIdx, values } = sparse;
  for (let k = 0; k < values.length; k++) {
    out[rowIdx[k] * cols + colIdx[k]] += values[k];
  }
}

// ============================================================================
// Public entry point
// ============================================================================

export async function loadFunctionalDescriptor(shardData, location, name) {
  if (!location.descriptorManifest) {
    throw new Error(
      `[FunctionalDescriptor] location.descriptorManifest is required for tensor "${name}". ` +
      'The layer loader must parse the manifoldgguf manifest and attach it before calling loadTensorToGPU.'
    );
  }
  const manifest = assertFunctionalDescriptorManifest(
    location.descriptorManifest,
    `FUNCTIONAL_DESCRIPTOR tensor "${name}" descriptorManifest`
  );

  const shards = shardData?.descriptorShards;
  if (!shards || typeof shards.get !== 'function') {
    throw new Error(
      `[FunctionalDescriptor] shardData.descriptorShards (Map) is required for tensor "${name}". ` +
      'The shard manager must load all component shard files before dispatch.'
    );
  }

  const { components } = manifest;
  const [rows, cols] = normalizeShape2(
    manifest.padded_shape ?? manifest.slice_shape,
    'descriptor reconstruction shape'
  );
  const [cropRows, cropCols] = normalizeShape2(
    manifest.crop_shape ?? location.shape ?? manifest.slice_shape,
    'descriptor crop shape'
  );
  if (cropRows > rows || cropCols > cols) {
    throw new Error(
      `[FunctionalDescriptor] crop shape [${cropRows},${cropCols}] exceeds descriptor shape [${rows},${cols}] for "${name}".`
    );
  }
  const descriptorBytes = getDescriptorShardBytes(shards);
  const denseF16Bytes = cropRows * cropCols * 2;
  assertRuntimeProofGates(manifest, name, descriptorBytes, denseF16Bytes);

  log.debug('FunctionalDescriptor', `Reconstructing "${name}" [${rows}x${cols}]`);

  const reconstructed = new Float32Array(rows * cols);

  // PRNG substrate
  const prngSpec = components.prng_substrate;
  const prng = buildPRNG(prngSpec, rows, cols);
  for (let i = 0; i < reconstructed.length; i++) reconstructed[i] += prng[i];

  // Kronecker sum
  const kronSpec = components.kronecker_sum;
  const kronShard = shards.get(kronSpec.shard_file);
  if (!kronShard) {
    throw new Error(`[FunctionalDescriptor] Kron shard "${kronSpec.shard_file}" not found in descriptorShards for "${name}".`);
  }
  const kronFactors = deserializeKronecker(
    kronShard instanceof Uint8Array ? kronShard : new Uint8Array(kronShard)
  );
  const kron = reconstructKronecker(kronFactors, rows, cols);
  for (let i = 0; i < reconstructed.length; i++) reconstructed[i] += kron[i];

  // SIREN INR
  const sirenSpec = components.coordinate_inr;
  const sirenShard = shards.get(sirenSpec.shard_file);
  if (!sirenShard) {
    throw new Error(`[FunctionalDescriptor] SIREN shard "${sirenSpec.shard_file}" not found in descriptorShards for "${name}".`);
  }
  const sirenLayers = deserializeSIREN(
    sirenShard instanceof Uint8Array ? sirenShard : new Uint8Array(sirenShard)
  );
  const siren = reconstructSIREN(sirenLayers, rows, cols);
  for (let i = 0; i < reconstructed.length; i++) reconstructed[i] += siren[i];

  // Sparse residuals
  const sparseSpec = components.sparse_outliers;
  const sparseShard = shards.get(sparseSpec.shard_file);
  if (!sparseShard) {
    throw new Error(`[FunctionalDescriptor] Sparse shard "${sparseSpec.shard_file}" not found in descriptorShards for "${name}".`);
  }
  const sparse = deserializeSparse(
    sparseShard instanceof Uint8Array ? sparseShard : new Uint8Array(sparseShard)
  );
  applySparse(reconstructed, sparse, cols);

  log.debug('FunctionalDescriptor', `"${name}" reconstructed, uploading to GPU`);
  const out = cropMatrix(reconstructed, rows, cols, cropRows, cropCols);

  // Upload as dense F32 weight buffer
  const gpuBuf = acquireBuffer(out.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  uploadData(gpuBuf, out);

  const data = createWeightBuffer(gpuBuf, 'f32', 'row', [cropRows, cropCols], name, null, {
    storageType: 'functional_descriptor',
    descriptorHash: manifest.descriptor_hash,
    descriptorBytes,
    denseF16Bytes,
    compressionRatio: descriptorBytes > 0 ? denseF16Bytes / descriptorBytes : null,
    proofStatus: manifest.proof_status ?? null,
    proofStatusGate: manifest.proof_status_gate ?? null,
    descriptorShape: [rows, cols],
    cropShape: [cropRows, cropCols],
  });
  return {
    data,
    allocatedBuffers: [gpuBuf],
  };
}
