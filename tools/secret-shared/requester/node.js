/**
 * Node bindings for the secret-shared requester: TCP transport, a file-backed weight source that reads only the
 * tensors the requester holds (the embedding row by row, never the whole table), SHAKE-256 from node:crypto, and
 * WebCrypto (globalThis.crypto) for AES-CTR, SHA-256 and the seeds.
 */
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import net from 'node:net';
import path from 'node:path';
import { float16ToFloat32 } from '../../../src/formats/float16.js';

/**
 * A framed byte stream over one TCP socket: write(bytes), read(n) -> Promise<Uint8Array>, close(). The entry is
 * { host, port } to dial an operator, or { accept: 'HOST:PORT' } to listen for one operator that dials in (an operator
 * host that refuses inbound connections runs `operator-gpu.js --connect`).
 */
export async function connect(entry) {
  if (entry.accept) {
    const at = entry.accept.lastIndexOf(':');
    const socket = await new Promise((resolve, reject) => {
      const server = net.createServer({ noDelay: true }, s => { server.close(); resolve(s); });
      server.once('error', reject);
      server.listen(Number(entry.accept.slice(at + 1)), entry.accept.slice(0, at));
      setTimeout(() => { server.close(); reject(Error(`no operator dialed ${entry.accept}`)); }, 60000).unref();
    });
    return stream(socket, true);
  }
  return stream(net.createConnection({ host: entry.host, port: entry.port, noDelay: true }), false);
}

function stream(socket, connected) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let have = 0, waiter = null, failed = null;
    const pump = () => {
      if (!waiter) return;
      if (failed) { const w = waiter; waiter = null; w.reject(failed); return; }
      if (have < waiter.n) return;
      const { n, resolve: done } = waiter;
      waiter = null;
      let out;
      if (chunks[0].length >= n) {
        out = chunks[0].subarray(0, n);
        if (chunks[0].length === n) chunks.shift(); else chunks[0] = chunks[0].subarray(n);
      } else {
        out = new Uint8Array(n);
        let off = 0;
        while (off < n) {
          const c = chunks[0], take = Math.min(c.length, n - off);
          out.set(c.subarray(0, take), off); off += take;
          if (take === c.length) chunks.shift(); else chunks[0] = c.subarray(take);
        }
      }
      have -= n;
      done(out);
    };
    socket.on('data', chunk => { chunks.push(new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.length)); have += chunk.length; pump(); });
    socket.on('error', error => { failed = error; pump(); reject(error); });
    socket.on('close', () => { failed ??= Error('operator closed the connection'); pump(); });
    const api = {
      write: bytes => socket.write(bytes),
      read: n => new Promise((res, rej) => { waiter = { n, resolve: res, reject: rej }; pump(); }),
      close: () => socket.destroy()
    };
    if (connected) { socket.setNoDelay(true); resolve(api); } else socket.once('connect', () => resolve(api));
  });
}

export function shake256(seed, round, size) {
  const counter = Buffer.alloc(8);
  counter.writeBigUInt64LE(BigInt(round));
  const out = createHash('shake256', { outputLength: size }).update(seed).update(counter).digest();
  return new Uint8Array(out.buffer, out.byteOffset, out.length);
}

export const nodeEnv = { connect, crypto: globalThis.crypto, shake256 };

/** Reads requester-held tensors from an RDRR directory; the embedding is read one row per lookup. */
export function fileSource(root) {
  const manifest = JSON.parse(fs.readFileSync(path.join(root, 'manifest.json'), 'utf8'));
  const fds = new Map();
  const fd = file => { if (!fds.has(file)) fds.set(file, fs.openSync(path.join(root, file), 'r')); return fds.get(file); };
  const spans = entry => (entry.spans ?? [{ shardIndex: entry.shard, offset: entry.offset, size: entry.size }])
    .map(s => ({ file: manifest.shards[s.shardIndex].filename, offset: s.offset, size: s.size }));
  const readRange = (entry, start, length, out = new Uint8Array(length)) => {
    let skip = start, off = 0;
    for (const s of spans(entry)) {
      if (skip >= s.size) { skip -= s.size; continue; }
      const take = Math.min(s.size - skip, length - off);
      fs.readSync(fd(s.file), out, off, take, s.offset + skip);
      off += take; skip = 0;
      if (off === length) break;
    }
    return out;
  };
  let heldTensors = 0, heldBytes = 0;
  const emb = manifest.tensors['model.language_model.embed_tokens.weight'];
  const hidden = emb.shape[1];
  const row = new Uint8Array(hidden * 2);
  const rowView = new DataView(row.buffer);
  return {
    manifest,
    tensor(name) {
      const entry = manifest.tensors[name];
      if (!entry) throw Error(`missing tensor ${name}`);
      heldTensors++; heldBytes += entry.size;
      return { dtype: entry.dtype, shape: entry.shape, bytes: readRange(entry, 0, entry.size) };
    },
    embeddingRow(id, dst) {
      if (emb.dtype !== 'F16') throw Error('embedding dtype');
      readRange(emb, id * hidden * 2, hidden * 2, row);
      for (let i = 0; i < hidden; i++) dst[i] = float16ToFloat32(rowView.getUint16(2 * i, true));
    },
    heldTensors: () => heldTensors,
    heldBytes: () => heldBytes,
    /** Any tensor's raw bytes, for plaintext controls and the layer check (never used on the secret-shared path). */
    raw(name) { const entry = manifest.tensors[name]; return { dtype: entry.dtype, shape: entry.shape, bytes: readRange(entry, 0, entry.size) }; },
    close() { for (const f of fds.values()) fs.closeSync(f); }
  };
}

/** Streams float32 rows into a .npy file and fixes its row count on close. */
export function npyWriter(file, width) {
  const fdOut = fs.openSync(file, 'w');
  const headerLength = 128;
  let rows = 0;
  const header = () => {
    let h = `{'descr': '<f4', 'fortran_order': False, 'shape': (${rows}, ${width}), }`;
    h = h.padEnd(headerLength - 10 - 1, ' ') + '\n';
    const out = Buffer.alloc(headerLength);
    out.write('\x93NUMPY', 0, 'latin1'); out[6] = 1; out[7] = 0; out.writeUInt16LE(headerLength - 10, 8); out.write(h, 10, 'latin1');
    return out;
  };
  fs.writeSync(fdOut, header());
  return {
    push(row) { fs.writeSync(fdOut, new Uint8Array(row.buffer, row.byteOffset, row.byteLength)); rows++; },
    close() { fs.writeSync(fdOut, header(), 0, headerLength, 0); fs.closeSync(fdOut); }
  };
}
