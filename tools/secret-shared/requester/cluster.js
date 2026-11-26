/**
 * The requester's side of the share-operator protocol (see ../gpu/operator-gpu.js): per
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
 *
 * Sharing off (`sharing: false`): the plaintext control of the research harness. Each stage opens only the operator at
 * `plaintextShare` (default 0) of its OPS entry, and every round sends that operator the encoded input x~ itself instead of a
 * share; the operator, unchanged, returns W~ x~ in the same wire format (the same returnShift). Same plan, cut, hosts,
 * encoding, framing, rounds and links as the secret-shared run, with one operator per stage and no sharing. **The
 * operator reads the plaintext activations**: it is a measurement control, never a protection.
 *
 * Stalls. Every reply the requester waits for has a deadline: `roundTimeoutMs` for a linear round and for the
 * session messages (hello, options, seed, step), and `closeTimeoutMs` for the closing message. A missed deadline
 * rejects with an `OperatorStallError` naming the stage and share(s) that did not reply, and cancels the session: every
 * operator connection is closed at once, so no operator is left waiting on a round that will never come and no later
 * message is sent. A close that misses its deadline does not throw; that operator's result is `{ timedOut: true }`.
 */
import { encode, encodeAndShare, recombine } from './ring.js';

const MAGIC = 0x31535352;
const enc = new TextEncoder(), dec = new TextDecoder();
export const DEFAULT_ROUND_TIMEOUT_MS = 300000;
export const DEFAULT_CLOSE_TIMEOUT_MS = 5000;

/** A missed reply deadline: `stalled` lists the { stage, share } that did not reply in time. */
export class OperatorStallError extends Error {
  constructor(message, { stalled, timeoutMs, what }) {
    super(message);
    this.name = 'OperatorStallError';
    this.code = 'operator-stalled';
    this.stalled = stalled;
    this.timeoutMs = timeoutMs;
    this.what = what;
  }
}

/** Rejects with `onTimeout()` unless `promise` settles within `ms`; the timer is cleared when it settles. */
export function withDeadline(promise, ms, onTimeout) {
  let timer;
  const deadline = new Promise((_, reject) => {
    timer = setTimeout(() => reject(onTimeout()), ms);
  });
  return Promise.race([promise, deadline]).finally(() => clearTimeout(timer));
}

export class OperatorConnection {
  constructor(stream, where = null) { this.stream = stream; this.where = where; this.sent = 0; this.received = 0; this.pending = 0; this.hello = null; this.caps = new Set(); this.closed = false; }

  /** Drops the connection without a closing message (cancellation); safe to repeat. */
  destroy() { if (!this.closed) { this.closed = true; try { this.stream.close(); } catch { /* already gone */ } } }

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

  /** Sends the closing message and waits up to `timeoutMs` for its reply; the connection is closed either way. */
  async close(op = 'close', timeoutMs = DEFAULT_CLOSE_TIMEOUT_MS) {
    if (this.closed) return { timedOut: false, cancelled: true };
    try {
      const [h] = await withDeadline(this.call({ op }), timeoutMs, () => stall(`${op}: no reply`, [this.where], timeoutMs, op));
      return h;
    } catch (error) {
      if (error?.code === 'operator-stalled') return { timedOut: true, stalled: this.where, timeoutMs };
      throw error;
    } finally { this.destroy(); }
  }
}

function stall(what, stalled, timeoutMs, kind) {
  const names = stalled.filter(Boolean).map(s => `stage ${s.stage} share ${s.share === 0 ? 'A' : 'B'}`).join(' and ');
  return new OperatorStallError(`${what} from ${names || 'an operator'} within ${timeoutMs} ms`, { stalled: stalled.filter(Boolean), timeoutMs, what: kind });
}

/** Connects, sends hello and waits up to `timeoutMs` for the whole exchange; a stalled operator is closed. */
export async function openOperator(connect, entry, token, { where = null, timeoutMs = DEFAULT_ROUND_TIMEOUT_MS } = {}) {
  let op = null;
  const opening = (async () => {
    const stream = await connect(entry);
    op = new OperatorConnection(stream, where);
    const [hello] = await op.call({ op: 'hello', token });
    op.hello = hello;
    for (const cap of hello.caps ?? []) op.caps.add(cap);
    return op;
  })();
  try { return await withDeadline(opening, timeoutMs, () => stall('hello: no reply', [where], timeoutMs, 'hello')); }
  catch (error) { op?.destroy(); opening.then(late => late.destroy(), () => {}); throw error; }
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
    c.roundTimeoutMs = options.roundTimeoutMs ?? DEFAULT_ROUND_TIMEOUT_MS;
    c.closeTimeoutMs = options.closeTimeoutMs ?? DEFAULT_CLOSE_TIMEOUT_MS;
    if (!(c.roundTimeoutMs > 0) || !(c.closeTimeoutMs > 0)) throw Error('roundTimeoutMs and closeTimeoutMs are positive milliseconds');
    c.cancelled = null;
    c.sharing = options.sharing !== false;
    c.plaintextShare = options.plaintextShare ?? 0;
    if (![0, 1].includes(c.plaintextShare)) throw Error('plaintextShare is 0 or 1');
    c.where = new Map();
    for (const stage of plan.stages) for (const group of stage.groups) {
      c.where.set(group.id, { stage: stage.index, in: group.in, out: group.tensors.reduce((s, t) => s + t.shape[0], 0) });
    }
    const opened = [];
    try {
      const entries = operators.stages.map(pair => c.sharing ? pair.map((o, share) => [o, share]) : [[pair[c.plaintextShare], c.plaintextShare]]);
      c.pairs = await Promise.all(entries.map((pair, stage) => Promise.all(pair.map(([o, share]) =>
        openOperator(env.connect, o, operators.token, { where: { stage, share }, timeoutMs: c.roundTimeoutMs }).then(op => { opened.push(op); return op; })))));
      await c.configure(options);
    } catch (error) {
      for (const op of opened) op.destroy();
      c.cancelled = { at: Date.now(), reason: String(error?.message ?? error) };
      throw error;
    }
    return c;
  }

  /** One request/reply with an operator under the round deadline (session messages). */
  callWithin(op, header) {
    return withDeadline(op.call(header), this.roundTimeoutMs, () => stall(`${header.op}: no reply`, [op.where], this.roundTimeoutMs, header.op));
  }

  /**
   * Cancels the session: closes every operator connection now, without a closing message, so the operators' sessions
   * end and nothing more is sent. Idempotent; the first reason is kept.
   */
  cancel(reason) {
    this.cancelled ??= { at: Date.now(), reason: String(reason?.message ?? reason ?? 'cancelled'), stalled: reason?.stalled ?? null };
    for (const op of this.pairs?.flat() ?? []) op.destroy();
    return this.cancelled;
  }

  async configure(options) {
    const c = this;
    c.pairs.forEach((pair, index) => pair.forEach((op, share) => {
      const h = op.hello;
      if (h.stage !== index || h.share !== (c.sharing ? share : c.plaintextShare) || h.fx !== c.fx || h.fw !== c.fw) throw Error(`operator hello does not match the plan: ${JSON.stringify(h)}`);
    }));
    c.seeded = c.sharing && options.seeded !== false;
    c.pipelineSteps = Boolean(options.pipelineSteps);
    c.settings = [];
    c.seeds = [];
    for (const pair of c.pairs) {
      const wantShift = options.returnShift ?? 0;
      const shift = wantShift && pair.every(op => op.caps.has('returnShift')) ? wantShift : 0;
      if (!c.sharing) {
        if (options.strict && shift !== wantShift) throw Error('an operator lacks a requested capability');
        if (shift) await c.callWithin(pair[0], { op: 'options', returnShift: shift });
        c.seeds.push(null);
        c.settings.push({ shift, width: (64 - shift) / 8, prg: 'none', key: null, zeros: null });
        continue;
      }
      let prg = options.prg ?? 'shake256';
      if (prg === 'aes256-ctr' && !pair[0].caps.has('prg:aes256-ctr')) prg = 'shake256';
      if (prg === 'shake256' && c.seeded && !c.env.shake256) throw Error('this environment has no SHAKE-256: use operators with the prg:aes256-ctr capability');
      if (options.strict && (shift !== wantShift || prg !== (options.prg ?? 'shake256'))) throw Error('an operator lacks a requested capability');
      if (shift || prg !== 'shake256') {
        await Promise.all(pair.map((op, share) => c.callWithin(op, { op: 'options', returnShift: shift, ...(share === 0 ? { prg } : {}) })));
      }
      const seed = randomBytes(c.env.crypto.getRandomValues.bind(c.env.crypto), 32);
      if (c.seeded) await c.callWithin(pair[0], { op: 'seed', seed: [...seed].map(b => b.toString(16).padStart(2, '0')).join('') });
      const key = prg === 'aes256-ctr' ? await c.env.crypto.subtle.importKey('raw', seed, 'AES-CTR', false, ['encrypt']) : null;
      c.seeds.push(seed);
      c.settings.push({ shift, width: (64 - shift) / 8, prg, key });
    }
    c.counter = 0;
    c.rounds = [];
    c.step = null;
    c.keepDigests = options.digests !== false;
    // Shares are uniform, so a digest of the first 4 KiB already shows that no share-B vector repeats; fullDigests
    // hashes every byte, as the earlier prototype requester did.
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

  /** x: Float32Array [n * in]. Returns Float32Array [n * out] = W x, computed only on shares. Any failure cancels the session. */
  async run(gid, x, n) {
    if (this.cancelled) throw Error(`the session was cancelled: ${this.cancelled.reason}`);
    try { return await this.round(gid, x, n); } catch (error) { this.cancel(error); throw error; }
  }

  async round(gid, x, n) {
    const { stage, in: width, out } = this.where.get(gid);
    if (x.length !== n * width) throw Error(`${gid}: input ${x.length}, want ${n} x ${width}`);
    if (!this.sharing) return this.plainRound(gid, x, n, stage, out);
    const pair = this.pairs[stage];
    const s = this.settings[stage];
    const t0 = performance.now();
    const counter = ++this.counter;
    const header = { op: 'linear', group: gid, n, round: counter, ...(this.step !== null ? { step: this.step } : {}) };
    const before = pair.map(op => [op.sent, op.received]);
    const replied = [0, 0];
    const size = n * width * 8;
    // Share A's request is a header only (seeded), independent of the data, so it goes first.
    // A reply, or the failure of the operator that owed it (attributed to its stage and share).
    const reply = i => pair[i].read().then(r => { replied[i] = performance.now(); return r; }, error => { error.operator ??= pair[i].where; throw error; });
    let a;
    if (this.seeded) {
      pair[0].write({ ...header, seeded: true });
      a = reply(0);
      a.catch(() => {});
    }
    const x0 = await this.share0(stage, counter, size);
    const x1 = new Uint32Array(n * width * 2);
    encodeAndShare(x, this.fx, new Uint32Array(x0.buffer, x0.byteOffset, size / 4), x1);
    if (!this.seeded) {
      pair[0].write({ ...header, seeded: false }, x0);
      a = reply(0);
      a.catch(() => {});
    }
    const shareB = new Uint8Array(x1.buffer);
    pair[1].write({ ...header, seeded: false }, shareB);
    const t1 = performance.now();
    const b = reply(1);
    const digestB = this.keepDigests ? digest(this.env.crypto.subtle, shareB, this.digestLimit) : null;
    // A round that stalls fails with the share(s) that did not reply, and cancels the session, instead of waiting forever.
    const both = Promise.all([a, b]);
    both.catch(() => {});
    const [[h0, y0], [h1, y1]] = await withDeadline(both, this.roundTimeoutMs, () => stall(`${gid} round ${counter}: no reply`,
      [0, 1].filter(i => !replied[i]).map(i => pair[i].where), this.roundTimeoutMs, 'linear'));
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

  /** Sharing off: x~ goes to the stage's one operator; its W~ x~ is decoded as a share added to zero. */
  async plainRound(gid, x, n, stage, out) {
    const [op] = this.pairs[stage];
    const s = this.settings[stage];
    const t0 = performance.now();
    const counter = ++this.counter;
    const before = [op.sent, op.received];
    const encoded = encode(x, this.fx);
    op.write({ op: 'linear', group: gid, n, round: counter, ...(this.step !== null ? { step: this.step } : {}), seeded: false }, new Uint8Array(encoded.buffer));
    const t1 = performance.now();
    const reply = op.read().catch(error => { error.operator ??= op.where; throw error; });
    const [h0, y0] = await withDeadline(reply, this.roundTimeoutMs, () => stall(`${gid} round ${counter}: no reply`, [op.where], this.roundTimeoutMs, 'linear'));
    const t2 = performance.now();
    if (!s.zeros || s.zeros.length < y0.length) s.zeros = new Uint8Array(y0.length);
    const y = new Float32Array(n * out);
    const maxAbs = recombine(y0, s.zeros, s.width, s.shift, this.fx, this.fw, y);
    if (!(maxAbs < 2 ** (62 - this.fx - this.fw))) throw Error(`${gid}: output outside the fixed-point range`);
    const t3 = performance.now();
    this.rounds.push({ group: gid, stage, n, counter, shareBSha256: null, digestBytes: 0, encodeMs: t1 - t0, wireMs: t2 - t1, decodeMs: t3 - t2,
      replyMs: [t2 - t1, 0], operatorMs: [h0.computeNs / 1e6, 0], up: [op.sent - before[0], 0], down: [op.received - before[1], 0] });
    return y;
  }

  /** Tells every operator that `step` ended, so each reports it before its output is acknowledged. */
  async endStep(step) {
    if (this.cancelled) throw Error(`the session was cancelled: ${this.cancelled.reason}`);
    const ops = this.pairs.flat();
    if (this.pipelineSteps) { for (const op of ops) { op.write({ op: 'step', step }); op.pending++; } return; }
    try { await Promise.all(ops.map(op => this.callWithin(op, { op: 'step', step }))); } catch (error) { this.cancel(error); throw error; }
  }

  bytes() { const ops = this.pairs.flat(); return { sent: ops.reduce((s, o) => s + o.sent, 0), received: ops.reduce((s, o) => s + o.received, 0) }; }

  /** Ends every session with `op` (close or exit). An operator that does not reply within closeTimeoutMs is dropped and
   *  reported as { timedOut: true }; after a cancellation nothing is sent and each result is { cancelled: true }. */
  async close(op = 'close') { return Promise.all(this.pairs.map(pair => Promise.all(pair.map(o => o.close(op, this.closeTimeoutMs))))); }

  describe() {
    return this.settings.map(s => ({ sharing: this.sharing ? 'on' : 'off', returnShift: s.shift, returnBytes: s.width,
      prg: this.sharing ? (this.seeded ? s.prg : 'explicit') : 'none', ...(this.sharing ? {} : { operatorShare: this.plaintextShare }) }));
  }
}
