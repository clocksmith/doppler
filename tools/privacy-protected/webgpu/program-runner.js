// Loads compiled GPU session programs (gpu_record.go: <base>.json + <base>.bin) and runs their steps on an HeDevice.
// The encrypted KV cache stays on the GPU between steps, in a session state map (name -> ciphertext register).

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { outLevel } from './he-device.js';
import { prepareProgramBlob } from './rns-tables.js';

export const GPU_PROGRAM_SCHEMA = 'doppler.protected-gpu-program/v1';

export function loadProgram(base) {
  const json = readFileSync(`${base}.json`);
  const program = JSON.parse(json.toString('utf8'));
  const raw = readFileSync(`${base}.bin`);
  const digest = `sha256:${createHash('sha256').update(raw).digest('hex')}`;
  if (program.schema !== GPU_PROGRAM_SCHEMA || !program.steps?.length || raw.byteLength !== 4 * program.blobWords
    || digest !== program.blobDigest) {
    throw new Error(`program ${base} is malformed or its constants do not match their digest`);
  }
  const blob = new Uint32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
  const identity = `sha256:${createHash('sha256').update(json).update(raw).digest('hex')}`;
  return { program, blob, identity };
}

export function uploadProgram(he, program, blob) {
  return he.upload(prepareProgramBlob(program, blob.slice()), 'program-constants');
}

// Runs one step. `inputs` are the step's one-hot registers; `state` holds the session's KV cache and receives this
// step's new entries. With `timeLayers`, the queue is drained at every layer boundary, so each layer's time splits
// into host encoding and GPU wait.
export async function runStep(he, program, step, blobBuf, inputs, state,
  { timeLayers = true, flushEvery = 0, now = () => performance.now() } = {}) {
  he.flushEvery = flushEvery;
  if (inputs.length !== step.inputs.length) throw new Error(`step takes ${step.inputs.length} inputs`);
  const regs = new Map();
  const owned = new Set();
  step.inputs.forEach((id, i) => {
    const r = program.regs[id];
    if (inputs[i].deg !== r.deg || inputs[i].level !== r.level) throw new Error(`input ${i} has the wrong shape`);
    regs.set(id, inputs[i]);
    owned.add(id);
  });
  for (const s of step.stateIn ?? []) {
    const ct = state.get(s.name);
    if (!ct) throw new Error(`session state has no ${s.name}`);
    regs.set(s.reg, ct);
    owned.add(s.reg);
  }
  const keep = new Set((step.stateOut ?? []).map((s) => s.reg));
  const layerAt = new Map((step.layers ?? []).map((layer) => [layer.first, layer]));
  const layerEnd = new Map((step.layers ?? []).map((layer) => [layer.last, layer]));
  const layers = [];
  let open = null;
  const t0 = now();
  const d0 = he.stats.dispatches;
  const h0 = he.stats.hostMs;
  const w0 = he.stats.waitMs;
  for (let i = step.opFirst; i < step.opEnd; i++) {
    if (timeLayers && layerAt.has(i)) {
      await he.sync();
      open = { name: layerAt.get(i).name, start: now(), dispatches: he.stats.dispatches, host: he.stats.hostMs, wait: he.stats.waitMs };
    }
    const op = program.ops[i];
    const te = now();
    const out = he.allocCt(op.deg, outLevel(op));
    he.execOp(op, op.a >= 0 ? regs.get(op.a) : null, op.b >= 0 ? regs.get(op.b) : null, out, blobBuf);
    he.stats.hostMs += now() - te;
    regs.set(op.out, out);
    for (const f of op.free ?? []) {
      const ct = regs.get(f);
      if (ct && !owned.has(f) && !keep.has(f)) he.freeCt(ct);
      regs.delete(f);
    }
    if (timeLayers && open && layerEnd.has(i)) {
      await he.sync();
      layers.push({ name: open.name, ms: now() - open.start, dispatches: he.stats.dispatches - open.dispatches,
        hostMs: he.stats.hostMs - open.host, gpuWaitMs: he.stats.waitMs - open.wait });
      open = null;
    }
  }
  await he.sync();
  for (const s of step.stateOut ?? []) state.set(s.name, regs.get(s.reg));
  return { output: regs.get(step.output), ms: now() - t0, dispatches: he.stats.dispatches - d0,
    hostMs: he.stats.hostMs - h0, gpuWaitMs: he.stats.waitMs - w0, layers };
}
