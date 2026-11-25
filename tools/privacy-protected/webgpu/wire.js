// The protected-generation framing (protocol.go): u32 header length | header JSON | blobs. Each kind has a size
// ceiling checked before any blob byte is read; a kind the receiver is not waiting for ends the session unanswered.

import { createHash } from 'node:crypto';

export const MAX_HEADER = 64 << 10;
export const KIND_CEILING = {
  hello: 0, manifest: 0, key: 1 << 30, 'keys-held': 0, ready: 0, prefill: 8 * (64 << 20), decode: 8 * (64 << 20),
  result: 32 << 20, bye: 0,
  // Homomorphic split: the stage A -> stage B link (split.js). `hop` carries the carry ciphertexts of one step.
  'hop-hello': 0, 'hop-ready': 0, hop: 64 << 20,
};

export class Refusal extends Error {
  constructor(kind, want) {
    super(`refused unsolicited "${kind}" message while waiting for [${want.join(' ')}]; session ended`);
    this.kind = kind;
  }
}

// digest(b...) = sha256 over (u64 length || bytes) of each item, as in protocol.go.
export function digest(...items) {
  const h = createHash('sha256');
  for (const x of items) {
    const b = typeof x === 'string' ? Buffer.from(x) : x;
    const n = Buffer.alloc(8);
    n.writeBigUInt64LE(BigInt(b.length));
    h.update(n);
    h.update(b);
  }
  return `sha256:${h.digest('hex')}`;
}

export class Conn {
  constructor(socket, protocol) {
    this.socket = socket;
    this.protocol = protocol;
    this.sent = 0;
    this.received = 0;
    this.chunks = [];
    this.have = 0;
    this.waiter = null;
    this.ended = null;
    socket.on('data', (d) => {
      this.chunks.push(d);
      this.have += d.length;
      this.#wake();
    });
    socket.on('end', () => { this.ended = new Error('connection closed'); this.#wake(); });
    socket.on('error', (e) => { this.ended = e; this.#wake(); });
    socket.on('close', () => { this.ended = this.ended ?? new Error('connection closed'); this.#wake(); });
  }

  #wake() {
    const w = this.waiter;
    this.waiter = null;
    if (w) w();
  }

  async #read(n) {
    while (this.have < n) {
      if (this.ended) throw this.ended;
      await new Promise((resolve) => { this.waiter = resolve; });
    }
    const out = Buffer.allocUnsafe(n);
    let at = 0;
    while (at < n) {
      const c = this.chunks[0];
      const take = Math.min(c.length, n - at);
      c.copy(out, at, 0, take);
      at += take;
      if (take === c.length) this.chunks.shift();
      else this.chunks[0] = c.subarray(take);
    }
    this.have -= n;
    this.received += n;
    return out;
  }

  async send(header, ...blobs) {
    const h = { ...header, protocol: this.protocol, blobs: blobs.map((b) => b.length) };
    const raw = Buffer.from(JSON.stringify(h));
    const pre = Buffer.alloc(4);
    pre.writeUInt32LE(raw.length);
    const parts = [pre, raw, ...blobs];
    for (const p of parts) {
      if (this.ended) throw this.ended;
      if (!this.socket.write(p)) {
        // A peer that refuses a message closes the connection; the write must not wait for a drain that never comes.
        await new Promise((resolve) => {
          const done = () => {
            this.socket.off('drain', done);
            this.socket.off('close', done);
            resolve();
          };
          this.socket.once('drain', done);
          this.socket.once('close', done);
        });
      }
      this.sent += p.length;
    }
  }

  async recv(...want) {
    const pre = await this.#read(4);
    const n = pre.readUInt32LE(0);
    if (n === 0 || n > MAX_HEADER) throw new Error(`header length ${n} outside (0, ${MAX_HEADER}]`);
    const h = JSON.parse((await this.#read(n)).toString('utf8'));
    if (!want.includes(h.kind)) throw new Refusal(h.kind, want);
    if (h.protocol !== this.protocol) throw new Error(`protocol ${h.protocol}, expected ${this.protocol}`);
    const sizes = h.blobs ?? [];
    let total = 0;
    for (const b of sizes) {
      if (!Number.isInteger(b) || b < 0) throw new Error('invalid blob length');
      total += b;
    }
    if (total > KIND_CEILING[h.kind]) throw new Error(`${h.kind} carries ${total} bytes, above its ${KIND_CEILING[h.kind]} ceiling`);
    const blobs = [];
    for (const b of sizes) blobs.push(await this.#read(b));
    return { header: h, blobs };
  }
}
