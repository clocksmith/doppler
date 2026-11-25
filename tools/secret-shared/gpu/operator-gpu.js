#!/usr/bin/env node
/**
 * Share operator on the GPU: the Go operator's wire protocol, step reports and acknowledgment gate, with Q4_K
 * projections computed by the exact two-u32-limb WebGPU ring matvec (ring-matvec.js). The few F16/F32 projections
 * (DeltaNet in_proj_a and in_proj_b) run on the CPU with BigInt, exactly.
 *
 *   SS_TOKEN=... node operator-gpu.js --plan PLAN.json --model-dir DIR --stage S --share 0|1 --listen HOST:PORT [--gate]
 */
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import net from 'node:net';
import path from 'node:path';
import readline from 'node:readline';
import { parseArgs } from 'node:util';
import { openDevice, createRingMatVec } from './ring-matvec.js';

const MAGIC = 0x31535352;
const { values: args } = parseArgs({ options: { plan: { type: 'string' }, 'model-dir': { type: 'string' }, stage: { type: 'string' },
  share: { type: 'string' }, listen: { type: 'string' }, gate: { type: 'boolean' },
  threads: { type: 'string' } } });  // --threads is the CPU operator's flag; accepted so a supervisor can launch either
const token = process.env.SS_TOKEN;
if (!args.plan || !args['model-dir'] || !args.stage || !args.share || !args.listen || !token) {
  console.error('usage: operator-gpu.js --plan P --model-dir D --stage S --share 0|1 --listen HOST:PORT [--gate], with SS_TOKEN set');
  process.exit(2);
}
const stageIndex = Number(args.stage), share = Number(args.share);
const planBytes = fs.readFileSync(args.plan);
const plan = JSON.parse(planBytes);
const planSha = createHash('sha256').update(planBytes).digest('hex');
const stage = plan.stages.find(row => row.index === stageIndex);
if (!stage || plan.schema !== 'doppler.secret-shared-plan/v1') throw Error('stage not in plan');
const MASK = (1n << 64n) - 1n;
const f16 = bits => { const s = bits & 0x8000 ? -1 : 1, e = (bits >> 10) & 31, f = bits & 1023;
  return e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15); };
const fixed = v => BigInt.asUintN(64, BigInt(Math.round(v * 2 ** plan.fw)));

const started = Date.now();
const { device, adapterInfo } = await openDevice();
const ring = createRingMatVec(device, { fw: plan.fw });
const groups = new Map();
let weightBytes = 0;
for (const group of stage.groups) {
  const members = [];
  for (const t of group.tensors) {
    const data = Buffer.concat(t.spans.map(span => { const fd = fs.openSync(path.join(args['model-dir'], span.file), 'r');
      const buf = Buffer.alloc(span.size); fs.readSync(fd, buf, 0, span.size, span.offset); fs.closeSync(fd); return buf; }));
    if (createHash('sha256').update(data).digest('hex') !== t.sha256) throw Error(`${t.name}: digest does not match the plan`);
    const [rows, cols] = t.shape;
    weightBytes += data.length;
    if (t.dtype.startsWith('Q4_K')) members.push({ kind: 'q4k', rows, cols, gpu: ring.tensor(rows, cols, new Uint8Array(data.buffer, data.byteOffset, data.length)) });
    else {
      const dense = new Array(rows * cols);
      for (let i = 0; i < rows * cols; i++) dense[i] = fixed(t.dtype === 'F16' ? f16(data.readUInt16LE(2 * i)) : data.readFloatLE(4 * i));
      members.push({ kind: 'dense', rows, cols, dense });
    }
  }
  groups.set(group.id, { id: group.id, in: group.in, out: members.reduce((sum, m) => sum + m.rows, 0), members });
}
console.error(`stage ${stageIndex} share ${share}: ${groups.size} groups, ${weightBytes} weight bytes on ${JSON.stringify(adapterInfo)}, loaded in ${Date.now() - started} ms`);

const gateLines = args.gate ? readline.createInterface({ input: process.stdin })[Symbol.asyncIterator]() : null;
const peakRss = () => process.resourceUsage().maxRSS * 1024;

async function linear(group, n, input) {
  const words = new Uint32Array(input.buffer, input.byteOffset, input.byteLength / 4);
  const out = new Uint32Array(n * group.out * 2);
  // Q4_K members come first in every group (projection order in the plan), so one GPU submit covers them all.
  const gpuMembers = group.members.filter(m => m.kind === 'q4k');
  if (group.members.findIndex(m => m.kind !== 'q4k') !== -1 && group.members.findIndex(m => m.kind !== 'q4k') < gpuMembers.length) throw Error('dense member before a Q4_K member');
  const gpuRows = gpuMembers.reduce((sum, m) => sum + m.rows, 0);
  const t0 = performance.now();
  const y = gpuMembers.length ? await ring.run(gpuMembers.map(m => m.gpu), words, n, gpuRows) : new Uint32Array(0);
  const gpuMs = performance.now() - t0;
  for (let c = 0; c < n; c++) out.set(y.subarray(2 * c * gpuRows, 2 * (c + 1) * gpuRows), 2 * c * group.out);
  let offset = gpuRows;
  const x = new BigUint64Array(input.buffer, input.byteOffset, input.byteLength / 8);
  for (const m of group.members.filter(member => member.kind === 'dense')) {
    for (let c = 0; c < n; c++) for (let r = 0; r < m.rows; r++) {
      let acc = 0n;
      for (let k = 0; k < m.cols; k++) acc += m.dense[r * m.cols + k] * x[c * m.cols + k];
      acc &= MASK;
      out[2 * (c * group.out + offset + r)] = Number(acc & 0xffffffffn);
      out[2 * (c * group.out + offset + r) + 1] = Number(acc >> 32n);
    }
    offset += m.rows;
  }
  return { out, gpuMs };
}

const prg = (seed, round, size) => { const counter = Buffer.alloc(8); counter.writeBigUInt64LE(BigInt(round));
  return createHash('shake256', { outputLength: size }).update(seed).update(counter).digest(); };

function frame(header, payload = Buffer.alloc(0)) {
  const js = Buffer.from(JSON.stringify(header));
  const head = Buffer.alloc(8); head.writeUInt32LE(MAGIC, 0); head.writeUInt32LE(js.length, 4);
  const len = Buffer.alloc(8); len.writeBigUInt64LE(BigInt(payload.length));
  return Buffer.concat([head, js, len, payload]);
}

const server = net.createServer({ noDelay: true }, socket => {
  let buffered = Buffer.alloc(0), authed = false, seed = null, busy = Promise.resolve();
  const stats = { rounds: 0, columns: 0, macs: 0, computeNs: 0, gpuNs: 0, bytesIn: 0, bytesOut: 0 };
  const st = { step: -1, authorized: false, rounds: 0, columns: 0, macs: 0, bytesOut: 0, transcript: createHash('sha256') };
  const report = () => {
    if (st.step >= 0 && st.rounds > 0) {
      process.stdout.write(`STEP ${JSON.stringify({ step: st.step, rounds: st.rounds, columns: st.columns, macs: st.macs, bytesOut: st.bytesOut,
        transcript: st.transcript.copy().digest('hex') })}\n`);
    }
    Object.assign(st, { rounds: 0, columns: 0, macs: 0, bytesOut: 0 });
  };
  const begin = async next => {
    if (next !== st.step) { report(); st.step = next; st.authorized = false; }
    if (st.authorized || !gateLines) { st.authorized = true; return; }
    process.stdout.write(`WAIT ${next}\n`);
    const reply = await gateLines.next();
    if (reply.done || reply.value !== `GO ${next}`) throw Error(`the supervisor refused step ${next}`);
    st.authorized = true;
  };
  const handle = async (header, payload) => {
    stats.bytesIn += payload.length;
    if (!authed && header.op !== 'hello') throw Error('hello first');
    switch (header.op) {
      case 'hello':
        if (header.token !== token) throw Error('token');
        authed = true;
        return [{ stage: stageIndex, share, planSha256: planSha, modelId: plan.modelId, fx: plan.fx, fw: plan.fw, device: 'webgpu', adapter: adapterInfo,
          groups: [...groups.values()].map(g => ({ id: g.id, in: g.in, out: g.out })), weightBytes, peakRss: peakRss() }];
      case 'seed':
        if (share !== 0) throw Error('seed is for share 0 only');
        seed = Buffer.from(header.seed, 'hex');
        return [{ ok: true }];
      case 'linear': {
        const group = groups.get(header.group), n = header.n;
        if (!group || !(n > 0)) throw Error('group');
        const input = header.seeded ? prg(seed, header.round, n * group.in * 8) : payload;
        if (input.length !== n * group.in * 8) throw Error('payload size');
        if (Number.isInteger(header.step)) { if (header.step < st.step) throw Error('step went backwards'); await begin(header.step); }
        const t0 = process.hrtime.bigint();
        const aligned = Buffer.from(input);  // own, 8-byte aligned copy
        const { out, gpuMs } = await linear(group, n, aligned);
        const ns = Number(process.hrtime.bigint() - t0);
        const bytes = Buffer.from(out.buffer, out.byteOffset, out.byteLength);
        st.transcript.update(bytes);
        if (st.rounds === 0) st.columns = n;
        st.rounds++; st.macs += n * group.in * group.out; st.bytesOut += bytes.length;
        Object.assign(stats, { rounds: stats.rounds + 1, columns: stats.columns + n, macs: stats.macs + n * group.in * group.out,
          computeNs: stats.computeNs + ns, gpuNs: stats.gpuNs + Math.round(gpuMs * 1e6), bytesOut: stats.bytesOut + bytes.length });
        return [{ round: header.round, computeNs: ns, gpuMs }, bytes];
      }
      case 'step':
        if (header.step === st.step) report();
        return [{ ok: true }];
      case 'stats': return [{ stats, peakRss: peakRss(), weightBytes }];
      case 'close': report(); return [{ stats, peakRss: peakRss() }, null, 'close'];
      case 'exit': report(); return [{ stats, peakRss: peakRss() }, null, 'exit'];
      default: throw Error('op');
    }
  };
  socket.on('data', chunk => {
    buffered = Buffer.concat([buffered, chunk]);
    for (;;) {
      if (buffered.length < 8) return;
      const jsLen = buffered.readUInt32LE(4);
      if (buffered.length < 8 + jsLen + 8) return;
      const payloadLen = Number(buffered.readBigUInt64LE(8 + jsLen));
      if (buffered.length < 16 + jsLen + payloadLen) return;
      const header = JSON.parse(buffered.subarray(8, 8 + jsLen));
      const payload = buffered.subarray(16 + jsLen, 16 + jsLen + payloadLen);
      buffered = buffered.subarray(16 + jsLen + payloadLen);
      busy = busy.then(async () => {
        try {
          const [reply, body, after] = await handle(header, Buffer.from(payload));
          socket.write(frame(reply, body ?? undefined));
          if (after) { socket.end(); if (after === 'exit') process.exit(0); }
        } catch (error) { socket.write(frame({ error: String(error?.message ?? error) })); socket.end(); }
      });
    }
  });
  socket.on('error', () => {});
});
const [host, port] = [args.listen.slice(0, args.listen.lastIndexOf(':')), Number(args.listen.slice(args.listen.lastIndexOf(':') + 1))];
server.listen(port, host, () => process.stdout.write(`READY ${host}:${port}\n`));
