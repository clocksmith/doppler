// Doppler-side TSIR boundary-point fixture writer.
//
// Builds frozen-Doppler-reference fixtures for the Doe Cerebras lane
// (see /home/x/deco/doe/config/doe-frozen-doppler-reference.schema.json).
// Captures handoff and TSIR boundary activation snapshots during a
// reference inference, writes them as `.npy` files keyed by layer
// index and probe-point name, and emits a manifest the Doe validator
// (bench/tools/validate_frozen_doppler_reference.py) can consume.
//
// Triggered when `operatorDiagnostics.tsirFixture` is set in the
// inference context. The CLI flag --tsir-fixture-dir on
// tools/run-program-bundle-reference.js plumbs it in.
//
// Stage name mapping (Doppler stage -> Doe TSIR boundary):
//   layer_in         -> pre_layer_input
//   post_input_norm  -> post_rmsnorm
//   linear_qkv_proj  -> post_qkv          (fused QKV path; single matmul)
//   q_proj+k_proj+v_proj -> post_qkv      (split QKV path; concatenated along
//                                          feature axis as Q∥K∥V at drain time
//                                          for Gemma 4 31B and similar models)
//   post_attn        -> post_attn
//   layer_out        -> post_ffn
// Additional debug-only stage mappings (post_qproj, post_kproj, post_vproj,
// post_qnorm, post_knorm, post_vnorm, attn_core_out, post_oproj, pre_ffn,
// ffn_mlp_out) emit per-stage .npy files at non-schema names; the Doe-side
// builder ignores them, but they are useful for parity bisection.

// Use dynamic import for `node:fs/promises` and `node:path` so this
// module loads cleanly in the browser bundle too (where Node builtins
// are unavailable). The fixture writer is a no-op in browser context;
// it only emits files when invoked from the Node-side harness.
import { readBufferSlice, acquireBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';
import { getDevice } from '../../../gpu/device.js';

let _nodeFsPromise = null;
async function getNodeFs() {
  if (_nodeFsPromise) return _nodeFsPromise;
  _nodeFsPromise = (async () => {
    try {
      const fs = await import('node:fs/promises');
      const path = await import('node:path');
      return { fs: fs.default ?? fs, path: path.default ?? path };
    } catch {
      return null;
    }
  })();
  return _nodeFsPromise;
}

const STAGE_TO_TSIR = {
  layer_in: 'pre_layer_input',
  post_input_norm: 'post_rmsnorm',
  // Two QKV-projection paths exist in Doppler:
  //   - Fused: `linear_qkv_proj` (linear-attention.js variants)
  //   - Split per-projection: `q_proj` (used by standard attention,
  //     including Gemma 4)
  // Map both to post_qkv. When both fire (which doesn't happen for
  // any current model), the q_proj snapshot wins because it's
  // overwritten last; that's fine — both name the same TSIR boundary
  // and hash equivalence checks tolerate the asymmetry of using just
  // the Q projection (the boundary is "post Q+K+V projection"; one
  // tensor captures that point in the per-block emit).
  linear_qkv_proj: 'post_qkv',
  q_proj: 'post_qproj',
  k_proj: 'post_kproj',
  v_proj: 'post_vproj',
  q_norm: 'post_qnorm',
  k_norm: 'post_knorm',
  v_norm: 'post_vnorm',
  attn_core_out: 'attn_core_out',
  attn_out: 'post_oproj',
  post_attn: 'post_attn',
  ffn_in: 'pre_ffn',
  ffn_out: 'ffn_mlp_out',
  layer_out: 'post_ffn',
};

export const TSIR_BOUNDARY_STAGES = Object.keys(STAGE_TO_TSIR);

export function mapStageToTsirBoundary(stage) {
  return STAGE_TO_TSIR[stage] ?? null;
}

// Track a per-layer q/k/v projection partial in the fixture-level Map and
// synthesize `post_qkv.npy` once all three are present. Called by both
// the synchronous write path of `maybeWriteFixtureSnapshot` AND the
// recorder-deferred drain path so split-q/k/v models always end up with
// the schema-binding `post_qkv` probe regardless of how the per-stage
// captures arrived. Returns a record describing the action (write,
// pending more partials, or write failure) so callers can append it to
// the fixture's records list.
async function trackQkvPartialAndMaybeWrite(tsirFixture, stage, layerIdx, numTokens, hiddenSize, arr) {
  const qkvByLayer = (tsirFixture.qkvByLayer ??= new Map());
  let entry = qkvByLayer.get(layerIdx);
  if (!entry) {
    entry = { numTokens, q: null, k: null, v: null, qHidden: 0, kHidden: 0, vHidden: 0 };
    qkvByLayer.set(layerIdx, entry);
  }
  if (stage === 'q_proj') { entry.q = arr; entry.qHidden = hiddenSize; }
  else if (stage === 'k_proj') { entry.k = arr; entry.kHidden = hiddenSize; }
  else if (stage === 'v_proj') { entry.v = arr; entry.vHidden = hiddenSize; }
  if (!entry.q || !entry.k || !entry.v) {
    return { stage: 'q_proj+k_proj+v_proj', tsirStage: 'post_qkv', layerIdx, written: false, note: 'pending-other-projections' };
  }
  const node = await getNodeFs();
  if (!node) {
    return { stage: 'q_proj+k_proj+v_proj', tsirStage: 'post_qkv', layerIdx, written: false, note: 'node-fs-unavailable-in-browser' };
  }
  const { path } = node;
  const totalHidden = entry.qHidden + entry.kHidden + entry.vHidden;
  const merged = new Float32Array(entry.numTokens * totalHidden);
  for (let t = 0; t < entry.numTokens; t++) {
    const dstBase = t * totalHidden;
    merged.set(entry.q.subarray(t * entry.qHidden, (t + 1) * entry.qHidden), dstBase);
    merged.set(entry.k.subarray(t * entry.kHidden, (t + 1) * entry.kHidden), dstBase + entry.qHidden);
    merged.set(entry.v.subarray(t * entry.vHidden, (t + 1) * entry.vHidden), dstBase + entry.qHidden + entry.kHidden);
  }
  const layerDir = path.join(tsirFixture.dir, `layer_${layerIdx}`);
  const filePath = path.join(layerDir, 'post_qkv.npy');
  qkvByLayer.delete(layerIdx);
  try {
    const info = await writeNpyF32(filePath, [entry.numTokens, totalHidden], merged);
    return {
      stage: 'q_proj+k_proj+v_proj',
      tsirStage: 'post_qkv',
      layerIdx,
      filePath,
      shape: [entry.numTokens, totalHidden],
      dtype: 'float32',
      ...info,
      written: true,
      note: `synthesized from q/k/v concat (qHidden=${entry.qHidden}, kHidden=${entry.kHidden}, vHidden=${entry.vHidden})`,
    };
  } catch (e) {
    return { stage: 'q_proj+k_proj+v_proj', tsirStage: 'post_qkv', layerIdx, written: false, note: `qkv-concat-write-failed: ${e?.message ?? e}` };
  }
}

// numpy .npy v1.0 writer for f32. Shape is a flat array of ints.
// data is a Float32Array (already-decoded; caller handles f16->f32).
export async function writeNpyF32(filePath, shape, data) {
  const node = await getNodeFs();
  if (!node) throw new Error('writeNpyF32 requires Node fs/path (not available in browser)');
  const { fs, path } = node;
  const magic = Uint8Array.of(0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59); // \x93NUMPY
  const ver = Uint8Array.of(0x01, 0x00); // major=1 minor=0
  const shapeStr = shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(', ')})`;
  let header = `{'descr': '<f4', 'fortran_order': False, 'shape': ${shapeStr}, }`;
  // Pad header so the data starts at a 64-byte boundary.
  // Total header = magic(6) + ver(2) + headerLen(2) + headerText
  const baseLen = 6 + 2 + 2 + header.length + 1; // +1 for trailing \n
  const padding = (64 - (baseLen % 64)) % 64;
  header = header + ' '.repeat(padding) + '\n';
  const headerBytes = new TextEncoder().encode(header);
  if (headerBytes.length > 0xFFFF) throw new Error('npy header too large for v1.0');
  const headerLen = Uint8Array.of(headerBytes.length & 0xFF, (headerBytes.length >> 8) & 0xFF);

  const payload = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const out = new Uint8Array(magic.length + ver.length + headerLen.length + headerBytes.length + payload.length);
  let off = 0;
  out.set(magic, off); off += magic.length;
  out.set(ver, off); off += ver.length;
  out.set(headerLen, off); off += headerLen.length;
  out.set(headerBytes, off); off += headerBytes.length;
  out.set(payload, off);

  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, out);
  return { byteLength: out.length, payloadByteLength: payload.length };
}

// f16 -> f32 conversion for buffer readback.
function f16ToF32(h) {
  const sign = (h >> 15) & 0x1;
  const exp = (h >> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0) return (sign ? -1 : 1) * Math.pow(2, -14) * (mant / 1024);
  if (exp === 31) return mant === 0 ? (sign ? -Infinity : Infinity) : NaN;
  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + mant / 1024);
}

function decodeBuffer(arrayBuffer, dtype) {
  if (dtype === 'f16') {
    const u16 = new Uint16Array(arrayBuffer);
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) out[i] = f16ToF32(u16[i]);
    return out;
  }
  return new Float32Array(arrayBuffer);
}

// Read a Doppler buffer (CPU Float32Array OR WebGPU GPUBuffer) into
// a Float32Array of the expected element count. Caller passes the
// dtype so f16 can be widened. For GPU buffers we use the buffer's
// MAP_READ availability; if mapping fails we return null and the
// fixture write is silently skipped (caller logs).
async function readBufferToFloat32(buffer, dtype, expectedElems) {
  if (buffer instanceof Float32Array) {
    if (buffer.length < expectedElems) return null;
    return buffer.slice(0, expectedElems);
  }
  // WebGPU GPUBuffer path: use the buffer-pool's existing readback
  // helper so staging buffers + alignment + readback gates all match
  // the rest of Doppler's instrumentation.
  if (buffer && typeof buffer.size === 'number') {
    try {
      const bytesPerElem = dtype === 'f16' ? 2 : 4;
      const bytes = expectedElems * bytesPerElem;
      const arrayBuffer = await readBufferSlice(buffer, 0, bytes);
      if (!arrayBuffer || arrayBuffer.byteLength === 0) return null;
      return decodeBuffer(arrayBuffer, dtype);
    } catch (e) {
      return null;
    }
  }
  return null;
}

// Record a probe snapshot if the (stage, layerIdx) pair matches the
// fixture writer's filter. Returns a record describing what was
// written, or null if skipped. Errors are caught and returned as a
// note on the record so the caller can log without crashing the run.
export async function maybeWriteFixtureSnapshot(stage, buffer, options) {
  const { tsirFixture, layerIdx, numTokens, hiddenSize, dtype = 'f32', recorder } = options;
  if (!tsirFixture || !tsirFixture.dir) return null;
  const tsirStage = mapStageToTsirBoundary(stage);
  if (!tsirStage) return null;
  const layerFilter = tsirFixture.layerFilter ?? null;
  if (Array.isArray(layerFilter) && !layerFilter.includes(layerIdx)) return null;
  // Skip decode-step writes (numTokens === 1) so prefill snapshots aren't
  // overwritten by single-token decode passes. Fixture semantics expect a
  // multi-token prefill capture.
  if (tsirFixture.prefillOnly !== false && numTokens === 1) return null;
  // When a prefill recorder is in-flight, the buffer doesn't have the
  // recorded ops applied yet — the staging copy would return zeros from
  // the pool. Defer the snapshot until the recorder has been submitted
  // by queuing the metadata; drainPendingTsirReads() processes the queue
  // after recorder.submitAndWait().
  if (recorder) {
    // Copy the in-flight buffer's bytes to a freshly-acquired carry buffer
    // using the same recorder's encoder. When the recorder submits, the
    // carry buffer holds the values at this probe point. Then drain after
    // submitAndWait().
    const expectedBytes = numTokens * hiddenSize * (dtype === 'f16' ? 2 : 4);
    const alignedBytes = Math.ceil(expectedBytes / 4) * 4;
    if (!buffer || typeof buffer.size !== 'number') return null;
    if (typeof recorder.getEncoder !== 'function') return null;
    const carry = acquireBuffer(
      Math.max(alignedBytes, 256),
      typeof buffer.usage === 'number' ? buffer.usage : undefined,
      `tsir_carry_${tsirStage}_L${layerIdx}`
    );
    try {
      recorder.getEncoder().copyBufferToBuffer(buffer, 0, carry, 0, alignedBytes);
    } catch (e) {
      releaseBuffer(carry);
      return { stage, tsirStage, layerIdx, written: false, note: `carry-copy-failed: ${e?.message ?? e}` };
    }
    const pending = (tsirFixture.pendingReads ??= []);
    pending.push({ stage, tsirStage, layerIdx, numTokens, hiddenSize, dtype, carry, alignedBytes });
    return { stage, tsirStage, layerIdx, written: false, note: 'pending-recorder-submit' };
  }

  const expectedElems = numTokens * hiddenSize;
  const data = await readBufferToFloat32(buffer, dtype, expectedElems);
  if (!data) {
    return {
      stage,
      tsirStage,
      layerIdx,
      written: false,
      note: 'buffer-readback-failed',
    };
  }

  const node = await getNodeFs();
  if (!node) {
    return {
      stage,
      tsirStage,
      layerIdx,
      written: false,
      note: 'node-fs-unavailable-in-browser',
    };
  }
  const { path } = node;
  const layerDir = path.join(tsirFixture.dir, `layer_${layerIdx}`);
  const filePath = path.join(layerDir, `${tsirStage}.npy`);
  let perStageRecord;
  try {
    const info = await writeNpyF32(filePath, [numTokens, hiddenSize], data);
    perStageRecord = {
      stage,
      tsirStage,
      layerIdx,
      filePath,
      shape: [numTokens, hiddenSize],
      dtype: 'float32',
      ...info,
      written: true,
    };
  } catch (e) {
    return {
      stage,
      tsirStage,
      layerIdx,
      written: false,
      note: `write-failed: ${e?.message ?? e}`,
    };
  }
  // Synthesize post_qkv.npy on the fly when split-q/k/v projections are
  // captured via the synchronous path (recorder undefined). Models with
  // a fused linear_qkv_proj path emit post_qkv directly; this branch
  // handles the split-projection case (Gemma 4 31B etc.).
  if (stage === 'q_proj' || stage === 'k_proj' || stage === 'v_proj') {
    const qkvRecord = await trackQkvPartialAndMaybeWrite(
      tsirFixture, stage, layerIdx, numTokens, hiddenSize, data,
    );
    if (qkvRecord && qkvRecord.written && Array.isArray(tsirFixture.records)) {
      tsirFixture.records.push(qkvRecord);
    }
  }
  return perStageRecord;
}


// Drain pending TSIR snapshots that were queued during a prefill recorder
// pass. Call this after `recorder.submitAndWait()` so the carry buffers
// hold valid recorded values. Releases each carry buffer after readback.
//
// After per-stage writes complete, this also synthesizes the schema-binding
// `post_qkv.npy` for layers that have all three of `post_qproj`,
// `post_kproj`, `post_vproj` captured. The Doe schema
// (config/doe-frozen-doppler-reference.schema.json) requires the probe-point
// name `post_qkv`. For models with split q_proj/k_proj/v_proj (Gemma 4 31B,
// etc.), we capture each individually for debug introspection AND emit the
// concatenated Q∥K∥V tensor along the feature axis as `post_qkv.npy` so the
// Doe-side builder, validator, and splice receipts can consume it. The
// fused-QKV path (`linear_qkv_proj`) writes to `post_qkv.npy` directly and
// is not re-synthesized.
export async function drainPendingTsirReads(tsirFixture) {
  if (!tsirFixture || !tsirFixture.dir) return [];
  const pending = tsirFixture.pendingReads;
  if (!pending || pending.length === 0) return [];
  const node = await getNodeFs();
  const records = [];
  for (const item of pending) {
    const { stage, tsirStage, layerIdx, numTokens, hiddenSize, dtype, carry, alignedBytes } = item;
    let arr = null;
    let staging = null;
    let mapped = false;
    try {
      const device = getDevice();
      if (!device) {
        records.push({ stage, tsirStage, layerIdx, written: false, note: 'no-device' });
        continue;
      }
      staging = device.createBuffer({
        size: alignedBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        label: `tsir_drain_staging_${tsirStage}_L${layerIdx}`,
      });
      const enc = device.createCommandEncoder({ label: `tsir_drain_enc_L${layerIdx}` });
      enc.copyBufferToBuffer(carry, 0, staging, 0, alignedBytes);
      device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      mapped = true;
      const ab = staging.getMappedRange(0, alignedBytes).slice(0, numTokens * hiddenSize * (dtype === 'f16' ? 2 : 4));
      arr = decodeBuffer(ab, dtype);
    } catch (e) {
      records.push({ stage, tsirStage, layerIdx, written: false, note: `drain-read-failed: ${e?.message ?? e}` });
    } finally {
      if (mapped && staging) staging.unmap();
      if (staging) staging.destroy();
      releaseBuffer(carry);
    }
    if (!arr) continue;
    if (!node) {
      records.push({ stage, tsirStage, layerIdx, written: false, note: 'node-fs-unavailable-in-browser' });
      continue;
    }
    const { path } = node;
    const layerDir = path.join(tsirFixture.dir, `layer_${layerIdx}`);
    const filePath = path.join(layerDir, `${tsirStage}.npy`);
    try {
      const info = await writeNpyF32(filePath, [numTokens, hiddenSize], arr);
      records.push({ stage, tsirStage, layerIdx, filePath, shape: [numTokens, hiddenSize], dtype: 'float32', ...info, written: true });
    } catch (e) {
      records.push({ stage, tsirStage, layerIdx, written: false, note: `write-failed: ${e?.message ?? e}` });
      continue;
    }
    // Synthesize post_qkv.npy on the fly for split-q/k/v projection
    // models — same helper the synchronous write path calls so both
    // recorder-deferred and immediate captures end up with the
    // schema-binding probe.
    if (stage === 'q_proj' || stage === 'k_proj' || stage === 'v_proj') {
      const qkvRecord = await trackQkvPartialAndMaybeWrite(
        tsirFixture, stage, layerIdx, numTokens, hiddenSize, arr,
      );
      if (qkvRecord && qkvRecord.written) records.push(qkvRecord);
    }
  }
  tsirFixture.pendingReads = [];
  if (Array.isArray(tsirFixture.records)) {
    tsirFixture.records.push(...records);
  }
  return records;
}
