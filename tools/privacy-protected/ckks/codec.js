// The raw-u32/v0 wire format of the u32 profiles. One RNS residue is one little-endian u32, in Lattigo's NTT order,
// so the WebGPU evaluator uploads blobs to GPU buffers as they arrive. Every residue is checked to be below its
// modulus on decode.
//
// Ciphertext: "DHC0" | degree u32 | level u32 | logN u32 | logSlots u32 | 0 u32 | scale f64 |
//             (degree+1) x (level+1) x N residues (component, then limb, then coefficient).
// Key:        "DHK0" | kind u32 (0 relinearization, 1 Galois) | Galois element u64 | levelQ u32 | levelP u32 |
//             digits u32 | logN u32 | digits x 2 x ((levelQ+1) + (levelP+1)) x N residues, canonical
//             (digit, then component b/a, then Q limbs and P limbs, then coefficient).

import { LOG_SLOTS } from '../plan.js';

export const RAW_CT_MAGIC = 0x30434844; // "DHC0"
export const RAW_KEY_MAGIC = 0x304b4844; // "DHK0"
export const RAW_CT_HEADER = 32;
export const RAW_KEY_HEADER = 32;
export const KEY_KIND_RELIN = 0;
export const KEY_KIND_GALOIS = 1;

export function encodeCt(p, { deg, level, scale, body }) {
  if (body.length !== (deg + 1) * (level + 1) * p.n) throw new Error('ciphertext body has the wrong length');
  const out = Buffer.alloc(RAW_CT_HEADER + 4 * body.length);
  [RAW_CT_MAGIC, deg, level, p.logN, LOG_SLOTS, 0].forEach((w, i) => out.writeUInt32LE(w, 4 * i));
  out.writeDoubleLE(scale, 24);
  Buffer.from(body.buffer, body.byteOffset, body.byteLength).copy(out, RAW_CT_HEADER);
  return out;
}

function checkResidues(body, limbs, moduli, n) {
  for (let i = 0; i < limbs; i++) {
    const q = moduli[i % moduli.length];
    for (let j = 0; j < n; j++) {
      if (body[i * n + j] >= q) throw new Error(`residue ${j} of limb ${i % moduli.length} is not below its modulus`);
    }
  }
}

// decodeCt: anyScale accepts a zero scale field (the split's hop ciphertexts carry none).
export function decodeCt(p, raw, { anyScale = false } = {}) {
  if (raw.length < RAW_CT_HEADER || raw.readUInt32LE(0) !== RAW_CT_MAGIC) throw new Error('not a raw-u32 ciphertext');
  const deg = raw.readUInt32LE(4);
  const level = raw.readUInt32LE(8);
  if (raw.readUInt32LE(12) !== p.logN || raw.readUInt32LE(16) !== LOG_SLOTS || deg < 1 || deg > 2 || level > p.maxLevel) {
    throw new Error('raw ciphertext header does not match the profile');
  }
  const scale = raw.readDoubleLE(24);
  if (!anyScale && (!(scale > 0) || !Number.isFinite(scale))) throw new Error('raw ciphertext scale is invalid');
  const words = (deg + 1) * (level + 1) * p.n;
  if (raw.length !== RAW_CT_HEADER + 4 * words) {
    throw new Error(`raw ciphertext has ${raw.length} bytes, expected ${RAW_CT_HEADER + 4 * words}`);
  }
  const body = new Uint32Array(words);
  Buffer.from(body.buffer).set(raw.subarray(RAW_CT_HEADER));
  const moduli = p.q.slice(0, level + 1);
  for (let c = 0; c <= deg; c++) checkResidues(body.subarray(c * (level + 1) * p.n), level + 1, moduli, p.n);
  return { deg, level, scale, body };
}

export function decodeKey(p, raw, wantKind) {
  if (raw.length < RAW_KEY_HEADER || raw.readUInt32LE(0) !== RAW_KEY_MAGIC) throw new Error('not a raw-u32 key');
  const kind = raw.readUInt32LE(4);
  const gal = Number(raw.readBigUInt64LE(8));
  if (kind !== wantKind || raw.readUInt32LE(16) !== p.maxLevel || raw.readUInt32LE(20) !== p.pCount - 1
    || raw.readUInt32LE(24) !== p.digits || raw.readUInt32LE(28) !== p.logN || (kind === 0) !== (gal === 0)) {
    throw new Error('raw key header does not match the profile');
  }
  const QP = p.qCount + p.pCount;
  const words = p.digits * 2 * QP * p.n;
  if (raw.length !== RAW_KEY_HEADER + 4 * words) throw new Error('raw key has the wrong length');
  const body = new Uint32Array(words);
  Buffer.from(body.buffer).set(raw.subarray(RAW_KEY_HEADER));
  checkResidues(body, p.digits * 2 * QP, [...p.q, ...p.p], p.n);
  return { kind, gal, body };
}
