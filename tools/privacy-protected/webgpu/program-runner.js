// Loads a compiled GPU program (gpu_record.go: <base>.json + <base>.bin) and runs it on an HeDevice.

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { outLevel } from './he-device.js';
import { prepareProgramBlob } from './rns-tables.js';

export const GPU_PROGRAM_SCHEMA = 'doppler.protected-gpu-program/v0';

export function loadProgram(base) {
  const program = JSON.parse(readFileSync(`${base}.json`, 'utf8'));
  const raw = readFileSync(`${base}.bin`);
  const digest = `sha256:${createHash('sha256').update(raw).digest('hex')}`;
  if (program.schema !== GPU_PROGRAM_SCHEMA || raw.byteLength !== 4 * program.blobWords || digest !== program.blobDigest) {
    throw new Error(`program ${base} is malformed or its constants do not match their digest`);
  }
  const blob = new Uint32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
  return { program, blob };
}

// The program's identity: what an evaluator states in its manifest and the requester could pin.
export function programDigest(base) {
  const h = createHash('sha256');
  h.update(readFileSync(`${base}.json`));
  h.update(readFileSync(`${base}.bin`));
  return `sha256:${h.digest('hex')}`;
}

export function uploadProgram(he, program, blob) {
  return he.upload(prepareProgramBlob(program, blob.slice()), 'program-constants');
}

// Runs the program on GPU registers `inputs`; returns the output register and per-layer timings. With `timeLayers`,
// the queue is drained at every layer boundary, so each layer's wall time is GPU execution plus its encoding.
export async function runProgram(he, program, blobBuf, inputs, { timeLayers = true, flushEvery = 0, now = () => performance.now() } = {}) {
  he.flushEvery = flushEvery;
  if (inputs.length !== program.inputs.length) throw new Error(`program takes ${program.inputs.length} inputs`);
  const regs = new Array(program.regs.length).fill(null);
  program.inputs.forEach((id, i) => {
    const r = program.regs[id];
    if (inputs[i].deg !== r.deg || inputs[i].level !== r.level) throw new Error(`input ${i} has the wrong shape`);
    regs[id] = inputs[i];
  });
  const layerAt = new Map(program.layers.map((layer) => [layer.first, layer]));
  const layerEnd = new Map(program.layers.map((layer) => [layer.last, layer]));
  const layers = [];
  let open = null;
  const t0 = now();
  const d0 = he.stats.dispatches;
  for (let i = 0; i < program.ops.length; i++) {
    if (timeLayers && layerAt.has(i)) {
      await he.sync();
      open = { name: layerAt.get(i).name, start: now(), dispatches: he.stats.dispatches, host: he.stats.hostMs, wait: he.stats.waitMs };
    }
    const op = program.ops[i];
    const te = now();
    const out = he.allocCt(op.deg, outLevel(op));
    he.execOp(op, op.a >= 0 ? regs[op.a] : null, op.b >= 0 ? regs[op.b] : null, out, blobBuf);
    he.stats.hostMs += now() - te;
    regs[op.out] = out;
    for (const f of op.free ?? []) {
      if (regs[f] && !program.inputs.includes(f)) he.freeCt(regs[f]);
      regs[f] = null;
    }
    if (timeLayers && open && layerEnd.has(i)) {
      await he.sync();
      layers.push({ name: open.name, ms: now() - open.start, dispatches: he.stats.dispatches - open.dispatches,
        hostMs: he.stats.hostMs - open.host, gpuWaitMs: he.stats.waitMs - open.wait });
      open = null;
    }
  }
  await he.sync();
  return { output: regs[program.output], ms: now() - t0, dispatches: he.stats.dispatches - d0, layers };
}
