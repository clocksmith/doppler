// The WebGPU evaluator: the same session protocol as `pp evaluator` (evaluator.go), with every model layer computed
// on ciphertexts by WGSL kernels. It receives public evaluation keys and ciphertexts only. It has no code path that
// creates or accepts the requester's private key or turns a ciphertext into a value; it answers nothing but the steps
// of the pinned schedule; its log and report carry sizes, timings and digests, never values.
//
//   node tools/privacy-protected/webgpu/evaluator.js --program <base> [--listen 127.0.0.1:0] [--ready-file f]
//        [--report r.json] [--misbehave ask-decrypt|tamper]
//
// <base>.json/.bin is a program compiled by `pp gpu-compile` from the public fixture (prefill of N prompt tokens).

import { writeFileSync } from 'node:fs';
import { createServer } from 'node:net';
import { arch, platform } from 'node:os';
import { parseArgs } from 'node:util';
import { HeDevice } from './he-device.js';
import { openDevice } from './gpu.js';
import { loadProgram, programDigest, runProgram, uploadProgram } from './program-runner.js';
import { Conn, digest } from './wire.js';

const ROTATIONS = [1, 2, 4, 8, 16, 32];
const EPOCH = /^[a-f0-9]{32}$/;

const { values } = parseArgs({
  options: {
    program: { type: 'string' },
    listen: { type: 'string', default: '127.0.0.1:0' },
    'ready-file': { type: 'string' },
    report: { type: 'string' },
    misbehave: { type: 'string', default: '' },
    // per-layer: drain the queue at each layer boundary (per-layer host and GPU times); overlap: submit every 256
    // dispatches so encoding overlaps GPU execution (lowest latency, no per-layer split).
    timing: { type: 'string', default: 'per-layer' },
  },
});
if (!values.program) throw new Error('--program <base> is required');

const log = (msg) => console.error(`[evaluator webgpu ${platform()}/${arch()}] ${msg}`);
const rep = {
  schema: 'doppler.protected-evaluator-report/v0', backend: 'webgpu', arch: `${platform()}/${arch()}`, misbehave: values.misbehave,
  outcome: 'completed', steps: [],
};

async function main() {
  const t0 = performance.now();
  const { program, blob } = loadProgram(values.program);
  rep.profile = program.profile;
  rep.programDigest = programDigest(values.program);
  const gpu = await openDevice();
  rep.provider = gpu.provider;
  rep.adapter = gpu.adapter;
  const he = new HeDevice(gpu.device, program);
  const blobBuf = uploadProgram(he, program, blob);
  await he.sync();
  rep.setupSeconds = (performance.now() - t0) / 1000;
  log(`program ${rep.programDigest} (${program.ops.length} ops, ${program.positions}-token prefill) on ${JSON.stringify(gpu.adapter)}`);

  const [host, port] = values.listen.split(/:(?=[^:]*$)/);
  const server = createServer();
  await new Promise((resolve) => server.listen(Number(port), host, resolve));
  const addr = server.address();
  if (values['ready-file']) writeFileSync(values['ready-file'], `${addr.address}:${addr.port}`, { mode: 0o600 });
  const socket = await new Promise((resolve) => server.once('connection', resolve));
  server.close();
  const c = new Conn(socket, program.protocol);
  try {
    await session(c, he, program, blobBuf);
  } finally {
    rep.bytesReceived = c.received;
    rep.bytesSent = c.sent;
    rep.gpu = {
      dispatches: he.stats.dispatches, submits: he.stats.submits, peakBufferBytes: he.stats.peakBytes,
      evaluationKeyBytesResident: he.stats.keyBytes, uploadBytes: he.stats.uploadBytes, readbackBytes: he.stats.readbackBytes,
    };
    socket.destroy();
    await gpu.close();
  }
}

async function session(c, he, program, blobBuf) {
  const { header: h } = await c.recv('hello');
  const f = h.fields ?? {};
  if (f.profile !== program.profile) throw new Error(`requester pinned profile ${f.profile}, this evaluator runs ${program.profile}`);
  if (f.model !== program.model) throw new Error(`requester pinned model ${f.model}, this evaluator serves ${program.model}`);
  if (f.promptTokens !== program.positions || f.generatedTokens !== 1) {
    throw new Error(`this evaluator holds only the ${program.positions}-token prefill program (one generated token)`);
  }
  if (!EPOCH.test(f.keyEpoch ?? '') || !(f.keyEpochExpires > Date.now() / 1000)) throw new Error('key epoch missing or expired');
  const session = h.session;
  rep.keyEpoch = f.keyEpoch;
  await c.send({ kind: 'manifest', session, fields: {
    profile: program.profile, profileDigest: program.profileDigest, model: program.model, rotations: ROTATIONS,
    inputLevel: program.q.length - 1, logSlots: program.logSlots, heldKeysDigest: '', backend: 'webgpu', programDigest: rep.programDigest,
  } });
  // One relinearization key, then one Galois key per pinned rotation; this evaluator keeps no key store.
  const tk = performance.now();
  const kd = [];
  const gals = [];
  for (let i = 0; i < 1 + ROTATIONS.length; i++) {
    const { blobs } = await c.recv('key');
    const key = he.loadKey(blobs[0]);
    if ((i === 0) !== (key.kind === 0)) throw new Error('keys arrived out of order');
    if (key.kind === 1) gals.push(key.gal);
    kd.push(digest(blobs[0]));
  }
  if (!program.galois.every((g) => gals.includes(g))) {
    throw new Error('Galois keys do not cover the program');
  }
  await he.sync();
  const keysDigest = digest(...kd);
  rep.keysDigest = keysDigest;
  rep.keyLoadSeconds = (performance.now() - tk) / 1000;
  await c.send({ kind: 'ready', session, fields: { keysDigest } });
  log(`ready: keys digest ${keysDigest}, ${(he.stats.keyBytes / 2 ** 20).toFixed(1)} MiB of keys resident`);

  for (let seq = 1; ; seq++) {
    const { header: s, blobs } = await c.recv('prefill', 'decode', 'bye');
    if (s.kind === 'bye') {
      log(`bye after ${seq - 1} steps`);
      return;
    }
    if (s.session !== session || s.seq !== seq) throw new Error('step binding mismatch');
    if (s.kind !== 'prefill' || seq !== 1 || blobs.length !== program.positions) {
      throw new Error(`unexpected ${s.kind} at step ${seq}: only the ${program.positions}-token prefill is compiled`);
    }
    const ts = performance.now();
    const inputs = blobs.map((b) => {
      const ct = he.uploadCt(b, { degree: 1, level: program.q.length - 1 });
      if (ct.scale !== program.inputScale) throw new Error('input ciphertext does not have the pinned scale');
      return ct;
    });
    const overlap = values.timing === 'overlap';
    const run = await runProgram(he, program, blobBuf, inputs, { timeLayers: !overlap, flushEvery: overlap ? 256 : 0 });
    let out = Buffer.from(await he.readCt(run.output, program.outputScale));
    const compute = (performance.now() - ts) / 1000;
    for (const x of inputs) he.freeCt(x);
    he.freeCt(run.output);
    rep.steps.push({ seq, kind: s.kind, position: program.positions - 1, computeSeconds: compute, gpuProgramMs: run.ms,
      dispatches: run.dispatches, timing: values.timing, hostMs: he.stats.hostMs, gpuWaitMs: he.stats.waitMs, layers: run.layers,
      resultDigest: digest(out), resultBytes: out.length });
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
    await c.send({ kind: 'result', session, seq, fields: { computeSeconds: compute, outputLevel: program.outputLevel, position: program.positions - 1 } }, out);
    log(`step ${seq} (${s.kind}): ${compute.toFixed(3)} s, program ${run.ms.toFixed(1)} ms, ${run.dispatches} dispatches, output level ${program.outputLevel}`);
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
