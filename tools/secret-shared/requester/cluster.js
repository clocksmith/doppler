/**
 * The requester's side of the share-operator protocol (see ../operator/main.go and ../gpu/operator-gpu.js): per
 * projection group, encode the input in Z_(2^64), split it into a PRG share for operator A and x~ - x0 for operator B,
 * send both, and recombine the two returned output shares.
 *
 * Environment-neutral: the transport (`connect(entry)` -> { write(bytes), read(n) -> Promise<Uint8Array>, close() },
 * where entry is an operator entry of OPS.json: { host, port }, or { accept: 'HOST:PORT' } for an operator that dials in) and the crypto (`shake256(seed, round, size)`, optional; WebCrypto `subtle` and `getRandomValues`) are
 * injected, so the same code runs in Node and in a browser. A browser has no SHAKE-256, so it needs operators that
 * advertise the `prg:aes256-ctr` capability.
 *
 * Options, each used only when every operator it touches advertises it in its hello's `caps`:
 *   returnShift  operators return bits returnShift..63 of their shares (40-bit returns at 24)
 *   prg          'aes256-ctr' (AES-256-CTR, key = seed, IV = round as u64 big-endian || 8 zero bytes) or 'shake256'
 *   pipelineSteps  the end-of-step message's reply is read before the next round's reply instead of awaited
 */
import { encodeAndShare, recombine } from './ring.js';

const MAGIC = 0x31535352;
const enc = new TextEncoder(), dec = new TextDecoder();

export class OperatorConnection {
  constructor(stream) { this.stream = stream; this.sent = 0; this.received = 0; this.pending = 0; this.hello = null; this.caps = new Set(); }

  write(header, payload) {
    const js = enc.encode(JSON.stringify(header));
    const len = payload ? payload.byteLength : 0;
    const frame = new Uint8Array(16 + js.length + (len < 65536 ? len : 0));
    const view = new DataView(frame.buffer);
    view.setUint32(0, MAGIC, true); view.setUint32(4, js.length, true);
    frame.set(js, 8);
    view.setUint32(8 + js.length, len % 4294967296, true); view.setUint32(12 + js.length, Math.floor(len / 4294967296), true);
    if (len && len < 65536) frame.set(new Uint8Array(payload.buffer, payload.byteOffset, len), 16 + js.length);
    this.stream.write(frame);
    if (len >= 65536) this.stream.write(new Uint8Array(payload.buffer, payload.byteOffset, len));
    this.sent += 16 + js.length + len;
  }

  async readFrame() {
    const head = await this.stream.read(8);
    const view = new DataView(head.buffer, head.byteOffset, 8);
    if (view.getUint32(0, true) !== MAGIC) throw Error('bad magic');
    const jsLen = view.getUint32(4, true);
    const rest = await this.stream.read(jsLen + 8);
    const header = JSON.parse(dec.decode(rest.subarray(0, jsLen)));
    const lv = new DataView(rest.buffer, rest.byteOffset + jsLen, 8);
    const plen = lv.getUint32(0, true) + lv.getUint32(4, true) * 4294967296;
    const payload = plen ? await this.stream.read(plen) : new Uint8Array(0);
    this.received += 16 + jsLen + plen;
    if (header.error) throw Error(`operator error: ${header.error}`);
    return [header, payload];
  }

  async read() {
    while (this.pending > 0) { this.pending--; await this.readFrame(); }
    return this.readFrame();
  }

  async call(header, payload) { this.write(header, payload); return this.read(); }

  async close(op = 'close') {
    try { const [h] = await this.call({ op }); return h; } finally { this.stream.close(); }
  }
}

export async function openOperator(connect, entry, token) {
  const op = new OperatorConnection(await connect(entry));
  const [hello] = await op.call({ op: 'hello', token });
  op.hello = hello;
  for (const cap of hello.caps ?? []) op.caps.add(cap);
  return op;
}

async function aesCtr(subtle, key, round, size) {
  const counter = new Uint8Array(16);
  new DataView(counter.buffer).setBigUint64(0, BigInt(round), false);
  const out = await subtle.encrypt({ name: 'AES-CTR', counter, length: 64 }, key, new Uint8Array(size));
  return new Uint8Array(out);
}

function randomBytes(getRandomValues, size) {
  const out = new Uint8Array(size);
  for (let i = 0; i < size; i += 65536) getRandomValues(out.subarray(i, Math.min(size, i + 65536)));
  return out;
}

/** Hex SHA-256 prefix of a share (of its first `limit` bytes, when set), for the freshness evidence (WebCrypto). */
async function digest(subtle, bytes, limit) {
  const h = new Uint8Array(await subtle.digest('SHA-256', limit && bytes.length > limit ? bytes.subarray(0, limit) : bytes));
  let s = '';
  for (let i = 0; i < 16; i++) s += h[i].toString(16).padStart(2, '0');
  return s;
}

export class SharedCluster {
  /**
   * plan: the doppler.secret-shared-plan/v1 JSON; operators: { token, stages: [[{host, port}, {host, port}], ...] }.
   * env: { connect, crypto: { subtle, getRandomValues }, shake256? }.
   */
  static async open(plan, operators, env, options = {}) {
    const c = new SharedCluster();
    c.fx = plan.fx; c.fw = plan.fw; c.env = env;
    c.where = new Map();
    for (const stage of plan.stages) for (const group of stage.groups) {
      c.where.set(group.id, { stage: stage.index, in: group.in, out: group.tensors.reduce((s, t) => s + t.shape[0], 0) });
    }
    c.pairs = await Promise.all(operators.stages.map(pair => Promise.all(pair.map(o => openOperator(env.connect, o, operators.token)))));
    c.pairs.forEach((pair, index) => pair.forEach((op, share) => {
      const h = op.hello;
      if (h.stage !== index || h.share !== share || h.fx !== c.fx || h.fw !== c.fw) throw Error(`operator hello does not match the plan: ${JSON.stringify(h)}`);
    }));
    c.seeded = options.seeded !== false;
    c.pipelineSteps = Boolean(options.pipelineSteps);
    c.settings = [];
    c.seeds = [];
    for (const pair of c.pairs) {
      const wantShift = options.returnShift ?? 0;
      const shift = wantShift && pair.every(op => op.caps.has('returnShift')) ? wantShift : 0;
      let prg = options.prg ?? 'shake256';
      if (prg === 'aes256-ctr' && !pair[0].caps.has('prg:aes256-ctr')) prg = 'shake256';
      if (prg === 'shake256' && c.seeded && !env.shake256) throw Error('this environment has no SHAKE-256: use operators with the prg:aes256-ctr capability');
      if (options.strict && (shift !== wantShift || prg !== (options.prg ?? 'shake256'))) throw Error('an operator lacks a requested capability');
      if (shift || prg !== 'shake256') {
        await Promise.all(pair.map((op, share) => op.call({ op: 'options', returnShift: shift, ...(share === 0 ? { prg } : {}) })));
      }
      const seed = randomBytes(env.crypto.getRandomValues.bind(env.crypto), 32);
      if (c.seeded) await pair[0].call({ op: 'seed', seed: [...seed].map(b => b.toString(16).padStart(2, '0')).join('') });
      const key = prg === 'aes256-ctr' ? await env.crypto.subtle.importKey('raw', seed, 'AES-CTR', false, ['encrypt']) : null;
      c.seeds.push(seed);
      c.settings.push({ shift, width: (64 - shift) / 8, prg, key });
    }
    c.counter = 0;
    c.rounds = [];
    c.step = null;
    c.keepDigests = options.digests !== false;
    // Shares are uniform, so a digest of the first 4 KiB already shows that no share-B vector repeats; fullDigests
    // hashes every byte, as requester.py did.
    c.digestLimit = options.fullDigests ? 0 : 4096;
    return c;
  }

  out(gid) { return this.where.get(gid).out; }

  async share0(stage, counter, size) {
    const s = this.settings[stage];
    if (!this.seeded) return randomBytes(this.env.crypto.getRandomValues.bind(this.env.crypto), size);
    if (s.prg === 'aes256-ctr') return aesCtr(this.env.crypto.subtle, s.key, counter, size);
    return this.env.shake256(this.seeds[stage], counter, size);
  }

  /** x: Float32Array [n * in]. Returns Float32Array [n * out] = W x, computed only on shares. */
  async run(gid, x, n) {
    const { stage, in: width, out } = this.where.get(gid);
    if (x.length !== n * width) throw Error(`${gid}: input ${x.length}, want ${n} x ${width}`);
    const pair = this.pairs[stage];
    const s = this.settings[stage];
    const t0 = performance.now();
    const counter = ++this.counter;
    const header = { op: 'linear', group: gid, n, round: counter, ...(this.step !== null ? { step: this.step } : {}) };
    const before = pair.map(op => [op.sent, op.received]);
    const replied = [0, 0];
    const size = n * width * 8;
    // Share A's request is a header only (seeded), independent of the data, so it goes first.
    let a;
    if (this.seeded) {
      pair[0].write({ ...header, seeded: true });
      a = pair[0].read().then(r => { replied[0] = performance.now(); return r; });
    }
    const x0 = await this.share0(stage, counter, size);
    const x1 = new Uint32Array(n * width * 2);
    encodeAndShare(x, this.fx, new Uint32Array(x0.buffer, x0.byteOffset, size / 4), x1);
    if (!this.seeded) {
      pair[0].write({ ...header, seeded: false }, x0);
      a = pair[0].read().then(r => { replied[0] = performance.now(); return r; });
    }
    const shareB = new Uint8Array(x1.buffer);
    pair[1].write({ ...header, seeded: false }, shareB);
    const t1 = performance.now();
    const b = pair[1].read().then(r => { replied[1] = performance.now(); return r; });
    const digestB = this.keepDigests ? digest(this.env.crypto.subtle, shareB, this.digestLimit) : null;
    const [[h0, y0], [h1, y1]] = await Promise.all([a, b]);
    const t2 = performance.now();
    const y = new Float32Array(n * out);
    const maxAbs = recombine(y0, y1, s.width, s.shift, this.fx, this.fw, y);
    if (!(maxAbs < 2 ** (62 - this.fx - this.fw))) throw Error(`${gid}: output outside the fixed-point range`);
    const t3 = performance.now();
    this.rounds.push({ group: gid, stage, n, counter, shareBSha256: digestB ? await digestB : null, digestBytes: Math.min(shareB.length, this.digestLimit || Infinity),
      encodeMs: t1 - t0, wireMs: t2 - t1, decodeMs: t3 - t2, replyMs: replied.map(r => r - t1),
      operatorMs: [h0.computeNs / 1e6, h1.computeNs / 1e6],
      up: pair.map((op, i) => op.sent - before[i][0]), down: pair.map((op, i) => op.received - before[i][1]) });
    return y;
  }

  /** Tells every operator that `step` ended, so each reports it before its output is acknowledged. */
  async endStep(step) {
    const ops = this.pairs.flat();
    if (this.pipelineSteps) { for (const op of ops) { op.write({ op: 'step', step }); op.pending++; } return; }
    await Promise.all(ops.map(op => op.call({ op: 'step', step })));
  }

  bytes() { const ops = this.pairs.flat(); return { sent: ops.reduce((s, o) => s + o.sent, 0), received: ops.reduce((s, o) => s + o.received, 0) }; }

  async close(op = 'close') { return Promise.all(this.pairs.map(pair => Promise.all(pair.map(o => o.close(op))))); }

  describe() { return this.settings.map(s => ({ returnShift: s.shift, returnBytes: s.width, prg: this.seeded ? s.prg : 'explicit' })); }
}
