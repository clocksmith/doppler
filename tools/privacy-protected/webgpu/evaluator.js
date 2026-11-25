// The WebGPU evaluator: the same session protocol as `pp evaluator` (evaluator.go), with every model layer computed
// on ciphertexts by WGSL kernels. It receives public evaluation keys and ciphertexts only. It has no code path that
// creates or accepts the requester's private key or turns a ciphertext into a value; it answers nothing but the steps
// of the pinned schedule; its log and report carry sizes, timings and digests, never values.
//
//   node tools/privacy-protected/webgpu/evaluator.js --program <base> [--program <base> ...] [--listen 127.0.0.1:0]
//        [--ready-file f] [--report r.json] [--state-dir d] [--sessions n] [--timing per-layer|overlap]
//        [--misbehave ask-decrypt|tamper]
//
// Each <base>.json/.bin is a session compiled by `pp gpu-compile` from the public fixture: a prefill step and decode
// steps. A request needs the prefill of its prompt length and a decode step for every later position it generates.
// The encrypted KV cache stays on the GPU for the session. Evaluation keys stay resident across sessions of one key
// epoch, and with --state-dir are kept on disk until the epoch's expiry, in the layout evaluator.go uses.

import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:net';
import { arch, platform } from 'node:os';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { HeDevice } from './he-device.js';
import { openDevice } from './gpu.js';
import { loadProgram, runStep, uploadProgram } from './program-runner.js';
import { Conn, digest } from './wire.js';

const ROTATIONS = [1, 2, 4, 8, 16, 32];
const EPOCH = /^[a-f0-9]{32}$/;
const MAX_KEY_BYTES = 1 << 30;

const { values } = parseArgs({
  options: {
    program: { type: 'string', multiple: true },
    listen: { type: 'string', default: '127.0.0.1:0' },
    'ready-file': { type: 'string' },
    report: { type: 'string' },
    'state-dir': { type: 'string' },
    sessions: { type: 'string', default: '1' },
    misbehave: { type: 'string', default: '' },
    // per-layer: drain the queue at each layer boundary (per-layer host and GPU times); overlap: submit every 256
    // dispatches so encoding overlaps GPU execution (lowest latency, no per-layer split).
    timing: { type: 'string', default: 'per-layer' },
  },
});
if (!values.program?.length) throw new Error('--program <base> is required');

const log = (msg) => console.error(`[evaluator webgpu ${platform()}/${arch()}] ${msg}`);
const rep = {
  schema: 'doppler.protected-evaluator-report/v1', backend: 'webgpu', arch: `${platform()}/${arch()}`, misbehave: values.misbehave,
  timing: values.timing, outcome: 'completed', sessions: [],
};

// --- key epochs on disk (same files as evaluator.go: epoch-<id>.json and epoch-<id>.keys) ---

function epochPath(dir, epoch) {
  return join(dir, `epoch-${epoch}`);
}

function pruneEpochs(dir) {
  for (const name of readdirSync(dir)) {
    if (!name.endsWith('.json')) continue;
    const base = join(dir, name.slice(0, -5));
    let expired = true;
    try {
      expired = Date.now() / 1000 >= JSON.parse(readFileSync(`${base}.json`, 'utf8')).Expires;
    } catch { /* unreadable: remove */ }
    if (expired) {
      rmSync(`${base}.json`, { force: true });
      rmSync(`${base}.keys`, { force: true });
    }
  }
}

function readKeyFile(path) {
  const raw = readFileSync(path);
  const out = [];
  for (let at = 0; at < raw.length;) {
    const n = Number(raw.readBigUInt64LE(at));
    if (n > MAX_KEY_BYTES || out.length > ROTATIONS.length) throw new Error('stored key file is malformed');
    out.push(raw.subarray(at + 8, at + 8 + n));
    at += 8 + n;
  }
  return out;
}

function writeKeyFile(path, blobs) {
  const parts = [];
  for (const b of blobs) {
    const n = Buffer.alloc(8);
    n.writeBigUInt64LE(BigInt(b.length));
    parts.push(n, b);
  }
  writeFileSync(path, Buffer.concat(parts), { mode: 0o600 });
}

// --- evaluator ---

async function main() {
  const t0 = performance.now();
  const gpu = await openDevice();
  rep.provider = gpu.provider;
  rep.adapter = gpu.adapter;
  const loaded = values.program.map((base) => loadProgram(base));
  const first = loaded[0].program;
  for (const { program } of loaded) {
    if (program.profileDigest !== first.profileDigest || program.model !== first.model) {
      throw new Error('programs disagree on profile or model');
    }
  }
  const he = new HeDevice(gpu.device, first);
  // Step index: kind:first:position -> the step and its program's constants.
  const steps = new Map();
  for (const { program, blob, identity } of loaded) {
    const blobBuf = uploadProgram(he, program, blob);
    for (const step of program.steps) {
      const key = `${step.kind}:${step.first}:${step.position}`;
      if (!steps.has(key)) steps.set(key, { program, step, blobBuf, identity });
    }
  }
  await he.sync();
  rep.programs = loaded.map((l) => l.identity);
  rep.steps = [...steps.keys()];
  rep.setupSeconds = (performance.now() - t0) / 1000;
  rep.programBufferBytes = he.stats.liveBytes;
  log(`programs ${rep.programs.join(', ')}: steps ${rep.steps.join(' ')} on ${JSON.stringify(gpu.adapter)}`);

  const stateDir = values['state-dir'];
  if (stateDir) {
    mkdirSync(stateDir, { recursive: true, mode: 0o700 });
    pruneEpochs(stateDir);
  }
  const resident = { epoch: '', digest: '' };
  const [host, port] = values.listen.split(/:(?=[^:]*$)/);
  const server = createServer();
  const sockets = [];
  server.on('connection', (s) => sockets.push(s));
  await new Promise((resolve) => server.listen(Number(port), host, resolve));
  const addr = server.address();
  if (values['ready-file']) writeFileSync(values['ready-file'], `${addr.address}:${addr.port}`, { mode: 0o600 });
  const limit = Number(values.sessions);
  try {
    for (let n = 0; limit === 0 || n < limit; n++) {
      while (sockets.length === 0) await new Promise((resolve) => server.once('connection', () => setImmediate(resolve)));
      const socket = sockets.shift();
      const c = new Conn(socket, first.protocol);
      const srep = { steps: [] };
      rep.sessions.push(srep);
      try {
        await session(c, he, first, steps, stateDir, resident, srep);
        srep.outcome = 'completed';
      } catch (e) {
        srep.outcome = 'failed';
        srep.error = e.message;
        log(`session error: ${e.message}`);
      } finally {
        srep.bytesReceived = c.received;
        srep.bytesSent = c.sent;
        socket.destroy();
      }
      if (values.misbehave) break;
    }
  } finally {
    server.close();
    rep.gpu = {
      dispatches: he.stats.dispatches, submits: he.stats.submits, peakBufferBytes: he.stats.peakBytes,
      evaluationKeyBytesResident: he.stats.keyBytes, uploadBytes: he.stats.uploadBytes, readbackBytes: he.stats.readbackBytes,
    };
    await gpu.close();
  }
}

async function session(c, he, first, steps, stateDir, resident, srep) {
  const { header: h } = await c.recv('hello');
  const f = h.fields ?? {};
  if (f.profile !== first.profile) throw new Error(`requester pinned profile ${f.profile}, this evaluator runs ${first.profile}`);
  if (f.model !== first.model) throw new Error(`requester pinned model ${f.model}, this evaluator serves ${first.model}`);
  const n = f.promptTokens;
  const g = f.generatedTokens;
  const plan = [`prefill:0:${n - 1}`];
  for (let p = n; p <= n + g - 2; p++) plan.push(`decode:${p}:${p}`);
  const missing = plan.filter((k) => !steps.has(k));
  if (!Number.isInteger(n) || !Number.isInteger(g) || n < 1 || g < 1 || missing.length) {
    throw new Error(`no compiled program for ${missing.join(' ') || 'this schedule'}`);
  }
  const epoch = f.keyEpoch ?? '';
  const expires = f.keyEpochExpires;
  if (!EPOCH.test(epoch) || !(expires > Date.now() / 1000)) throw new Error('key epoch missing or expired');
  const session = h.session;
  srep.keyEpoch = epoch;
  srep.schedule = { promptTokens: n, generatedTokens: g };
  let held = resident.epoch === epoch ? resident.digest : '';
  let stored = null;
  if (!held && stateDir && existsSync(`${epochPath(stateDir, epoch)}.json`)) {
    try {
      stored = JSON.parse(readFileSync(`${epochPath(stateDir, epoch)}.json`, 'utf8'));
      if (stored.ProfileDigest === first.profileDigest && stored.Expires > Date.now() / 1000) held = stored.KeysDigest;
    } catch { stored = null; }
  }
  await c.send({ kind: 'manifest', session, fields: {
    profile: first.profile, profileDigest: first.profileDigest, model: first.model, rotations: ROTATIONS,
    inputLevel: first.q.length - 1, logSlots: first.logSlots, heldKeysDigest: held, backend: 'webgpu',
  } });
  const tk = performance.now();
  const { header: kh, blobs: firstBlobs } = await c.recv('key', 'keys-held');
  let keyBlobs;
  if (kh.kind === 'keys-held') {
    if (!held) throw new Error('requester asked to reuse keys this evaluator does not hold');
    srep.keysFrom = resident.epoch === epoch ? 'gpu-resident' : 'store';
    keyBlobs = resident.epoch === epoch ? null : readKeyFile(`${epochPath(stateDir, epoch)}.keys`);
  } else {
    keyBlobs = [firstBlobs[0]];
    for (let i = 0; i < ROTATIONS.length; i++) keyBlobs.push((await c.recv('key')).blobs[0]);
    srep.keysFrom = 'upload';
  }
  if (keyBlobs) {
    if (keyBlobs.length !== 1 + ROTATIONS.length) throw new Error('wrong number of evaluation keys');
    const keysDigest = digest(...keyBlobs.map((b) => digest(b)));
    if (kh.kind === 'keys-held' && keysDigest !== held) throw new Error('stored keys do not match their recorded digest');
    he.clearKeys();
    const gals = [];
    keyBlobs.forEach((b, i) => {
      const key = he.loadKey(b);
      if ((i === 0) !== (key.kind === 0)) throw new Error('keys arrived out of order');
      if (key.kind === 1) gals.push(key.gal);
    });
    if (!first.galois.every((x) => gals.includes(x))) throw new Error('Galois keys do not cover the program');
    await he.sync();
    resident.epoch = epoch;
    resident.digest = keysDigest;
    if (kh.kind === 'key' && stateDir) {
      writeKeyFile(`${epochPath(stateDir, epoch)}.keys`, keyBlobs);
      writeFileSync(`${epochPath(stateDir, epoch)}.json`, JSON.stringify({
        Epoch: epoch, ProfileDigest: first.profileDigest, KeysDigest: keysDigest, Expires: expires,
      }), { mode: 0o600 });
    }
  }
  srep.keysDigest = resident.digest;
  srep.keyLoadSeconds = (performance.now() - tk) / 1000;
  await c.send({ kind: 'ready', session, fields: { keysDigest: resident.digest } });
  log(`ready: keys digest ${resident.digest} (${srep.keysFrom}), ${(he.stats.keyBytes / 2 ** 20).toFixed(1)} MiB of keys resident`);

  const state = new Map();
  try {
    for (let seq = 1; ; seq++) {
      const { header: s, blobs } = await c.recv('prefill', 'decode', 'bye');
      if (s.kind === 'bye') {
        log(`bye after ${seq - 1} steps`);
        return;
      }
      if (s.session !== session || s.seq !== seq || seq > plan.length) throw new Error('step binding mismatch');
      const entry = steps.get(plan[seq - 1]);
      if (s.kind !== entry.step.kind || blobs.length !== entry.step.inputs.length) {
        throw new Error(`unexpected ${s.kind} with ${blobs.length} inputs at step ${seq}`);
      }
      const ts = performance.now();
      const inputs = blobs.map((b) => {
        const ct = he.uploadCt(b, { degree: 1, level: first.q.length - 1 });
        if (ct.scale !== first.inputScale) throw new Error('input ciphertext does not have the pinned scale');
        return ct;
      });
      const overlap = values.timing === 'overlap';
      const run = await runStep(he, entry.program, entry.step, entry.blobBuf, inputs, state,
        { timeLayers: !overlap, flushEvery: overlap ? 256 : 0 });
      let out = Buffer.from(await he.readCt(run.output, entry.step.outputScale));
      const compute = (performance.now() - ts) / 1000;
      for (const x of inputs) he.freeCt(x);
      he.freeCt(run.output);
      srep.steps.push({ seq, kind: s.kind, position: entry.step.position, computeSeconds: compute, gpuProgramMs: run.ms,
        hostMs: run.hostMs, gpuWaitMs: run.gpuWaitMs, dispatches: run.dispatches, kvEntries: state.size,
        liveBufferBytes: he.stats.liveBytes, layers: run.layers, resultDigest: digest(out), resultBytes: out.length });
      if (values.misbehave === 'ask-decrypt') {
        log(`misbehave: sending a forged decrypt-request at step ${seq}`);
        await c.send({ kind: 'decrypt-request', session, seq }, blobs[0]);
        return;
      }
      if (values.misbehave === 'tamper') {
        out = Buffer.from(out);
        out[out.length >> 1] ^= 1;
        log(`misbehave: flipped one bit of the result at step ${seq}`);
      }
      await c.send({ kind: 'result', session, seq, fields: {
        computeSeconds: compute, outputLevel: entry.step.outputLevel, position: entry.step.position,
      } }, out);
      log(`step ${seq} (${s.kind}, position ${entry.step.position}): ${compute.toFixed(3)} s, program ${run.ms.toFixed(1)} ms, `
        + `${run.dispatches} dispatches, KV entries ${state.size}`);
    }
  } finally {
    for (const ct of state.values()) he.freeCt(ct);
    srep.peakBufferBytes = he.stats.peakBytes;
  }
}

try {
  await main();
} catch (e) {
  rep.outcome = 'failed';
  rep.error = e.message;
  log(`error: ${e.message}`);
  process.exitCode = 1;
} finally {
  if (values.report) writeFileSync(values.report, JSON.stringify(rep, null, 1), { mode: 0o600 });
}
