// The CPU backend of the protected evaluator: the HeDevice interface (he-device.js) over the JavaScript RNS executor
// (ckks/rns-exec.js), which computes every operation word for word as the WGSL kernels do. It is the second evaluator
// of the dual check and runs the route where no WebGPU adapter exists. Public material only: like the GPU device it
// has no key generation, encryption or decryption, only evaluation keys and ciphertexts.

import { decodeCt, decodeKey, encodeCt } from '../ckks/codec.js';
import { lookupProfile } from '../ckks/params.js';
import { RnsExec } from '../ckks/rns-exec.js';

export class CpuDevice {
  constructor(program) {
    const p = lookupProfile(program.profile);
    if (p.digest !== program.profileDigest || p.logN !== program.logN || JSON.stringify(p.q) !== JSON.stringify(program.q.map(Number))
      || JSON.stringify(p.p) !== JSON.stringify(program.p.map(Number))) {
      throw new Error('the program does not match its pinned profile');
    }
    this.cpu = true;
    this.p = p;
    this.exec = new RnsExec(p);
    this.flushEvery = 0;
    this.stats = { dispatches: 0, submits: 0, hostMs: 0, waitMs: 0, liveBytes: 0, peakBytes: 0, keyBytes: 0, uploadBytes: 0, readbackBytes: 0 };
  }

  upload(words) {
    return words;
  }

  allocCt(deg, level) {
    return { deg, level, body: null };
  }

  freeCt() {}

  execOp(op, a, b, out, blob) {
    const r = this.exec.run(op, a, b, blob);
    out.deg = r.deg;
    out.level = r.level;
    out.body = r.body;
    this.stats.dispatches++;
  }

  async sync() {}

  clearKeys() {
    this.exec.keys = { relin: null, galois: new Map() };
    this.stats.keyBytes = 0;
  }

  loadKey(raw) {
    const kind = raw.readUInt32LE(4);
    const key = decodeKey(this.p, raw, kind);
    this.exec.loadKey(key.kind, key.gal, key.body);
    this.stats.keyBytes += key.body.byteLength;
    return { kind: key.kind, gal: key.gal };
  }

  uploadCt(raw, { degree, level } = {}) {
    const ct = decodeCt(this.p, Buffer.from(raw.buffer, raw.byteOffset, raw.byteLength), { anyScale: true });
    if ((degree !== undefined && ct.deg !== degree) || (level !== undefined && ct.level !== level)) {
      throw new Error('raw ciphertext header does not match the pinned shape');
    }
    this.stats.uploadBytes += ct.body.byteLength;
    return ct;
  }

  async readCt(ct, scale) {
    this.stats.readbackBytes += ct.body.byteLength;
    return encodeCt(this.p, { deg: ct.deg, level: ct.level, scale, body: ct.body });
  }
}
