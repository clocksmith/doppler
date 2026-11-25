// Parallel key generation and encryption for the requester: worker threads of this same process (the secret key
// never leaves the process), each with its own CSPRNG stream. Evaluation keys are generated concurrently and handed
// out in protocol order; an encryption splits its limbs across the workers (the error polynomial is sampled once,
// here, and shared).

import { availableParallelism } from 'node:os';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { lookupProfile } from './params.js';
import { RandomStream, SecretKey, encryptLimbs, galoisKey, relinearizationKey, sampleGaussian } from './keys.js';
import { encodeCt } from './codec.js';

if (!isMainThread && workerData?.role === 'protected-key-worker') {
  const p = lookupProfile(workerData.profile);
  const sk = new SecretKey(p, new Int8Array(workerData.coeffs));
  const rng = new RandomStream();
  parentPort.on('message', (m) => {
    if (m.kind === 'key') {
      const raw = m.gal === 0 ? relinearizationKey(sk, rng) : galoisKey(sk, m.gal, rng);
      const ab = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
      parentPort.postMessage({ id: m.id, raw: ab }, [ab]);
    } else if (m.kind === 'encrypt') {
      const body = new Uint32Array(m.body);
      encryptLimbs(sk, m.compact, m.level, m.e, body, m.i0, m.i1, rng);
      parentPort.postMessage({ id: m.id });
    }
  });
}

export class KeyWorkers {
  // size: at most 4 workers by default. Each holds NTT tables and the secret in the NTT domain (about 80 MB at N = 2^16)
  // besides the key it is making, so 4 keeps the requester near 1 GB.
  constructor(sk, size = Math.max(1, Math.min(Number(process.env.PROTECTED_KEY_WORKERS) || 4, availableParallelism() - 1))) {
    this.sk = sk;
    this.next = 0;
    this.pending = new Map();
    this.workers = Array.from({ length: size }, () => {
      const w = new Worker(new URL(import.meta.url), {
        workerData: { role: 'protected-key-worker', profile: sk.p.id, coeffs: Array.from(sk.coeffs) },
      });
      w.on('message', (m) => {
        const r = this.pending.get(m.id);
        this.pending.delete(m.id);
        r?.resolve(m);
      });
      w.on('error', (e) => {
        for (const r of this.pending.values()) r.reject(e);
        this.pending.clear();
      });
      return w;
    });
    this.rr = 0;
  }

  call(msg, transfer = []) {
    const id = this.next++;
    const w = this.workers[this.rr++ % this.workers.length];
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      w.postMessage({ ...msg, id }, transfer);
    });
  }

  // Every evaluation key (relinearization, then one Galois key per pinned rotation), in that order. At most one key
  // per worker is in flight, so finished keys do not pile up (each is about 79 MB at N = 2^16).
  async *evaluationKeys() {
    const gals = [0, ...this.sk.p.galois];
    const inFlight = [];
    let next = 0;
    const start = () => {
      if (next < gals.length) inFlight.push(this.call({ kind: 'key', gal: gals[next++] }).then((m) => Buffer.from(m.raw)));
    };
    for (let i = 0; i < this.workers.length; i++) start();
    while (inFlight.length) {
      const key = await inFlight.shift();
      start();
      yield key;
    }
  }

  async encrypt(compact, level, scale) {
    const p = this.sk.p;
    const shared = new SharedArrayBuffer(4 * 2 * (level + 1) * p.n);
    const e = sampleGaussian(new RandomStream(1 << 16), p.n);
    const parts = Math.min(this.workers.length, level + 1);
    const per = Math.ceil((level + 1) / parts);
    const jobs = [];
    for (let i0 = 0; i0 <= level; i0 += per) {
      jobs.push(this.call({ kind: 'encrypt', compact, level, e, body: shared, i0, i1: Math.min(level + 1, i0 + per) }));
    }
    await Promise.all(jobs);
    return encodeCt(p, { deg: 1, level, scale, body: new Uint32Array(shared) });
  }

  async close() {
    await Promise.all(this.workers.map((w) => w.terminate()));
  }
}
