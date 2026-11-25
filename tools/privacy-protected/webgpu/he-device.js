// WebGPU execution of RNS-CKKS operations for the protected tiny-fixture evaluator.
//
// A ciphertext is one GPU buffer of (degree + 1) x (level + 1) x N u32 residues: component, then limb, then
// coefficient, in Lattigo's NTT order (the raw-u32/v0 wire layout, uploaded unchanged). Evaluation keys stay resident
// in Montgomery form. All work for a step is recorded into command buffers with one uniform arena and submitted in
// batches; nothing is read back until the result.
//
// Holds no secret material: there is no key generation, encryption or decryption here, only evaluation keys and
// ciphertexts.

import { readFileSync } from 'node:fs';
import {
  automorphismIndex, constantTables, digitsAt, modulusTable, twiddleTables,
} from './rns-tables.js';

const KERNEL_DIR = new URL('./kernels/', import.meta.url);
const WG = 256;
const UNIFORM_WORDS = 16;
const SLOT_BYTES = 256;
const ARENA_SLOTS = 8192;
const TILE = 2048;
const COMPACT_WORDS = 128;
const UNIFORM_FIELDS = ['n', 'log_n', 'limbs', 'mod_base', 'a_off', 'a_stride', 'b_off', 'b_stride', 'o_off', 'o_stride',
  'c_off', 'c_stride', 't', 'log_t', 'x0', 'x1'];

const MODULES = {
  he_ntt: ['ntt_fwd_global', 'ntt_fwd_local', 'ntt_inv_local', 'ntt_inv_global'],
  he_elementwise: ['ew_copy', 'ew_add', 'ew_sub', 'ew_neg', 'ew_smul', 'ew_sadd', 'ew_smac', 'ew_pmul', 'ew_padd',
    'ew_mmac', 'ew_mmul', 'ew_to_mont', 'ew_tensor', 'ew_sub_smul', 'ew_permute'],
  he_basis: ['conv_y', 'conv_sum', 'conv_copy1', 'rescale_prep'],
};

export const RAW_CT_MAGIC = 0x30434844;
export const RAW_KEY_MAGIC = 0x304b4844;
export const RAW_CT_HEADER = 32;
export const RAW_KEY_HEADER = 32;

export class HeDeviceError extends Error {
  constructor(code, message) {
    super(message);
    this.name = 'HeDeviceError';
    this.code = code;
  }
}

function ceilDiv(a, b) {
  return Math.ceil(a / b);
}

export class HeDevice {
  // params: { logN, logSlots, q: [], p: [], psi: [] } (numbers or numeric strings), from a compiled program.
  constructor(device, params) {
    this.device = device;
    this.logN = params.logN;
    this.n = 1 << params.logN;
    this.logSlots = params.logSlots;
    if (this.n < TILE * 2) {
      throw new HeDeviceError('HE_RING_TOO_SMALL', `ring degree ${this.n} is below the ${TILE * 2} the NTT tiles need`);
    }
    if (2 << params.logSlots !== COMPACT_WORDS) {
      throw new HeDeviceError('HE_LAYOUT', 'compact plaintexts assume 64 slots');
    }
    this.q = params.q.map(Number);
    this.p = params.p.map(Number);
    this.moduli = [...this.q, ...this.p];
    for (const m of this.moduli) {
      if (!(m > 2 ** 30 && m < 2 ** 32) || m % (2 * this.n) !== 1) {
        throw new HeDeviceError('HE_MODULUS', `modulus ${m} is not an NTT prime in (2^30, 2^32)`);
      }
    }
    this.qCount = this.q.length;
    this.pCount = this.p.length;
    this.stats = { dispatches: 0, submits: 0, hostMs: 0, waitMs: 0, liveBytes: 0, peakBytes: 0, keyBytes: 0, uploadBytes: 0, readbackBytes: 0 };
    this.bufferIds = new WeakMap();
    this.nextBufferId = 1;
    this.pool = new Map();
    this.bindGroups = new Map();
    this.commands = [];
    this.arenaWords = new Uint32Array(ARENA_SLOTS * (SLOT_BYTES / 4));
    this.arenaUsed = 0;
    this.keys = { relin: null, galois: new Map() };
    this.flushEvery = 0; // 0: submit when the uniform arena is full or at sync(); n: also every n dispatches
    this.permutations = new Map();
    this.#createPipelines();
    this.#uploadTables(params.psi);
    this.#allocateTemporaries();
  }

  // --- buffers ---

  #track(buffer, bytes) {
    this.bufferIds.set(buffer, this.nextBufferId++);
    this.stats.liveBytes += bytes;
    this.stats.peakBytes = Math.max(this.stats.peakBytes, this.stats.liveBytes);
    return buffer;
  }

  storage(words, label) {
    const size = Math.max(16, 4 * words);
    const buffer = this.device.createBuffer({
      label,
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    return this.#track(buffer, size);
  }

  upload(words, label) {
    const buffer = this.storage(words.length, label);
    this.device.queue.writeBuffer(buffer, 0, words.buffer, words.byteOffset, words.byteLength);
    return buffer;
  }

  destroy(buffer) {
    this.stats.liveBytes -= buffer.size;
    buffer.destroy();
  }

  // Ciphertext registers come from a pool keyed by size; command order on the queue makes reuse safe.
  allocCt(deg, level) {
    const words = (deg + 1) * (level + 1) * this.n;
    const free = this.pool.get(words);
    const buf = free && free.length ? free.pop() : this.storage(words, `ct-${deg}-${level}`);
    return { buf, deg, level, cs: (level + 1) * this.n };
  }

  freeCt(ct) {
    const words = (ct.deg + 1) * (ct.level + 1) * this.n;
    if (!this.pool.has(words)) this.pool.set(words, []);
    this.pool.get(words).push(ct.buf);
  }

  #createPipelines() {
    const d = this.device;
    const ro = { type: 'read-only-storage' };
    this.layout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: ro },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: ro },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: ro },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: ro },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: ro },
      ],
    });
    const pipelineLayout = d.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    const common = readFileSync(new URL('he_common.wgsl', KERNEL_DIR), 'utf8');
    this.pipelines = {};
    for (const [file, entries] of Object.entries(MODULES)) {
      const code = `${common}\n${readFileSync(new URL(`${file}.wgsl`, KERNEL_DIR), 'utf8')}`;
      const module = d.createShaderModule({ label: file, code });
      for (const entryPoint of entries) {
        this.pipelines[entryPoint] = d.createComputePipeline({
          label: entryPoint,
          layout: pipelineLayout,
          compute: { module, entryPoint, constants: { WORKGROUP_SIZE: WG } },
        });
      }
    }
    this.arena = d.createBuffer({ size: ARENA_SLOTS * SLOT_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.dummy = this.storage(4, 'dummy');
  }

  #uploadTables(psi) {
    this.modsBuf = this.upload(modulusTable(this.moduli, this.logN), 'moduli');
    const { fwd, inv } = twiddleTables(this.moduli, psi, this.logN);
    this.twFwd = this.upload(fwd, 'twiddles-forward');
    this.twInv = this.upload(inv, 'twiddles-inverse');
    this.tables = constantTables(this.qCount, this.pCount, this.moduli);
    this.tablesBuf = this.upload(this.tables.words, 'conversion-tables');
  }

  #allocateTemporaries() {
    const n = this.n;
    const Q = this.qCount;
    const QP = this.qCount + this.pCount;
    this.t = {
      inv: this.storage(Q * n, 'ks-inv'),
      ext: this.storage(QP * n, 'ks-ext'),
      yv: this.storage((Math.max(this.pCount, 2) + 1) * n, 'ks-yv'),
      acc: this.storage(2 * QP * n, 'ks-acc'),
      conv: this.storage(Q * n, 'ks-conv'),
      ks: this.storage(2 * Q * n, 'ks-out'),
      tmp: this.storage(2 * Q * n, 'rot-tmp'),
      last: this.storage(3 * n, 'rs-last'),
      rs: this.storage(3 * Q * n, 'rs-lifted'),
    };
  }

  // --- command recording ---

  #id(buffer) {
    return this.bufferIds.get(buffer);
  }

  #bindGroup(a, b, o, c, tw) {
    const key = `${this.#id(a)}|${this.#id(b)}|${this.#id(o)}|${this.#id(c)}|${this.#id(tw)}`;
    let bg = this.bindGroups.get(key);
    if (!bg) {
      bg = this.device.createBindGroup({
        layout: this.layout,
        entries: [
          { binding: 0, resource: { buffer: this.arena, size: UNIFORM_WORDS * 4 } },
          { binding: 1, resource: { buffer: a } },
          { binding: 2, resource: { buffer: b } },
          { binding: 3, resource: { buffer: o } },
          { binding: 4, resource: { buffer: c } },
          { binding: 5, resource: { buffer: this.modsBuf } },
          { binding: 6, resource: { buffer: tw } },
        ],
      });
      this.bindGroups.set(key, bg);
    }
    return bg;
  }

  dispatch(entry, bind, fields, x, y = 1) {
    if (x === 0 || y === 0) return;
    if (this.arenaUsed === ARENA_SLOTS) this.flush();
    const slot = this.arenaUsed++;
    const base = slot * (SLOT_BYTES / 4);
    const all = { n: this.n, log_n: this.logN, ...fields };
    for (let i = 0; i < UNIFORM_FIELDS.length; i++) this.arenaWords[base + i] = all[UNIFORM_FIELDS[i]] ?? 0;
    const bg = this.#bindGroup(bind.a ?? this.dummy, bind.b ?? this.dummy, bind.o, bind.c ?? this.dummy, bind.tw ?? this.dummy);
    this.commands.push({ pipeline: this.pipelines[entry], bg, offset: slot * SLOT_BYTES, x, y });
    this.stats.dispatches++;
    if (this.flushEvery && this.commands.length >= this.flushEvery) this.flush();
  }

  clear(buffer, offsetWords, words) {
    this.commands.push({ clear: buffer, offset: 4 * offsetWords, size: 4 * words });
  }

  flush() {
    if (this.commands.length === 0) return;
    this.device.queue.writeBuffer(this.arena, 0, this.arenaWords.buffer, 0, this.arenaUsed * SLOT_BYTES);
    const enc = this.device.createCommandEncoder();
    let pass = null;
    for (const cmd of this.commands) {
      if (cmd.clear) {
        if (pass) { pass.end(); pass = null; }
        enc.clearBuffer(cmd.clear, cmd.offset, cmd.size);
        continue;
      }
      if (!pass) pass = enc.beginComputePass();
      pass.setPipeline(cmd.pipeline);
      pass.setBindGroup(0, cmd.bg, [cmd.offset]);
      pass.dispatchWorkgroups(cmd.x, cmd.y);
    }
    if (pass) pass.end();
    this.device.queue.submit([enc.finish()]);
    this.stats.submits++;
    this.commands = [];
    this.arenaUsed = 0;
  }

  // Drains the queue. hostMs accumulates time spent encoding and submitting (in execOp and flush); waitMs the time
  // spent waiting for the GPU after the last submit.
  async sync() {
    const t0 = performance.now();
    this.flush();
    const t1 = performance.now();
    await this.device.queue.onSubmittedWorkDone();
    this.stats.hostMs += t1 - t0;
    this.stats.waitMs += performance.now() - t1;
  }

  // --- primitives ---

  // Elementwise kernel over `limbs` limbs and `comps` components.
  ew(entry, bind, f, comps = 1) {
    this.dispatch(entry, bind, {
      limbs: f.limbs, mod_base: f.modBase ?? 0, a_off: f.aOff ?? 0, a_stride: f.aStride ?? 0, b_off: f.bOff ?? 0,
      b_stride: f.bStride ?? 0, o_off: f.oOff ?? 0, o_stride: f.oStride ?? 0, c_off: f.cOff ?? 0, c_stride: f.cStride ?? 0,
      x0: f.x0 ?? 0, log_t: f.logT ?? 0,
    }, ceilDiv(f.limbs * this.n, WG), comps);
  }

  ntt(buf, off, limbs, modBase, fixed = false) {
    const n = this.n;
    for (let t = n >> 1, logT = this.logN - 1; t >= TILE; t >>= 1, logT--) {
      this.dispatch('ntt_fwd_global', { o: buf, tw: this.twFwd },
        { limbs, mod_base: modBase, o_off: off, t, log_t: logT, x1: fixed ? 1 : 0 }, ceilDiv(limbs * (n >> 1), WG));
    }
    this.dispatch('ntt_fwd_local', { o: buf, tw: this.twFwd }, { limbs, mod_base: modBase, o_off: off, x1: fixed ? 1 : 0 },
      limbs * (n / TILE));
  }

  intt(buf, off, limbs, modBase, fixed = false) {
    const n = this.n;
    this.dispatch('ntt_inv_local', { o: buf, tw: this.twInv }, { limbs, mod_base: modBase, o_off: off, x1: fixed ? 1 : 0 },
      limbs * (n / TILE));
    for (let t = TILE, logT = 11; t <= n >> 1; t <<= 1, logT++) {
      this.dispatch('ntt_inv_global', { o: buf, tw: this.twInv },
        { limbs, mod_base: modBase, o_off: off, t, log_t: logT, x0: t === n >> 1 ? 1 : 0, x1: fixed ? 1 : 0 },
        ceilDiv(limbs * (n >> 1), WG));
    }
  }

  // Hybrid key switch of polynomial (srcBuf, srcOff) at level l with gadget key `key`; result in this.t.ks
  // (component k at k * Q * N). Mirrors Lattigo's GadgetProduct: INTT, per digit basis extension to Q_l and P, NTT,
  // inner product with the key, then ModDown by P on both components.
  keySwitch(srcBuf, srcOff, l, key) {
    const n = this.n;
    const Q = this.qCount;
    const P = this.pCount;
    const QP = Q + P;
    const t = this.t;
    const tbl = this.tablesBuf;
    this.ew('ew_copy', { a: srcBuf, o: t.inv }, { limbs: l + 1, aOff: srcOff });
    this.intt(t.inv, 0, l + 1, 0);
    digitsAt(l, P).forEach(([s, e], d) => {
      const k = e - s;
      const targets = [[0, s], [e, l + 1], [Q, QP]].filter(([x, y]) => y > x);
      if (k >= 2) {
        const off = this.tables.offsets.digit.get(`${s}:${e}`);
        this.dispatch('conv_y', { a: t.inv, o: t.yv, c: tbl }, { a_off: s * n, mod_base: s, t: k, c_off: off }, ceilDiv(n, WG));
        for (const [x, y] of targets) {
          this.dispatch('conv_sum', { a: t.yv, o: t.ext, c: tbl },
            { limbs: y - x, mod_base: x, t: k, c_off: off, o_off: x * n }, ceilDiv((y - x) * n, WG));
        }
      } else {
        for (const [x, y] of targets) {
          this.dispatch('conv_copy1', { a: t.inv, o: t.ext },
            { limbs: y - x, mod_base: x, a_off: s * n, log_t: s, o_off: x * n }, ceilDiv((y - x) * n, WG));
        }
      }
      for (const [x, y] of targets) this.ntt(t.ext, x * n, y - x, x);
      this.ew('ew_copy', { a: srcBuf, o: t.ext }, { limbs: k, modBase: s, aOff: srcOff + s * n, oOff: s * n });
      const keyOff = d * 2 * QP * n;
      for (const [x, y] of [[0, l + 1], [Q, QP]]) {
        this.ew(d === 0 ? 'ew_mmul' : 'ew_mmac', { a: t.ext, o: t.acc, c: key.buf }, {
          limbs: y - x, modBase: x, aOff: x * n, aStride: 0, oOff: x * n, oStride: QP * n, cOff: keyOff + x * n, cStride: QP * n,
        }, 2);
      }
    });
    for (let cc = 0; cc < 2; cc++) {
      const base = cc * QP * n;
      this.intt(t.acc, base + Q * n, P, Q);
      this.dispatch('conv_y', { a: t.acc, o: t.yv, c: tbl }, { a_off: base + Q * n, mod_base: Q, t: P, c_off: this.tables.offsets.p },
        ceilDiv(n, WG));
      this.dispatch('conv_sum', { a: t.yv, o: t.conv, c: tbl }, { limbs: l + 1, mod_base: 0, t: P, c_off: this.tables.offsets.p },
        ceilDiv((l + 1) * n, WG));
      this.ntt(t.conv, 0, l + 1, 0);
      this.ew('ew_sub_smul', { a: t.acc, b: t.conv, o: t.ks, c: tbl },
        { limbs: l + 1, aOff: base, oOff: cc * Q * n, cOff: this.tables.offsets.pInv });
    }
  }

  relin(out, a, l) {
    if (!this.keys.relin) throw new HeDeviceError('HE_KEY_MISSING', 'no relinearization key loaded');
    this.keySwitch(a.buf, 2 * a.cs, l, this.keys.relin);
    this.ew('ew_add', { a: a.buf, b: this.t.ks, o: out.buf },
      { limbs: l + 1, aStride: a.cs, bStride: this.qCount * this.n, oStride: out.cs }, 2);
  }

  rotate(out, a, gal, l) {
    const key = this.keys.galois.get(gal);
    if (!key) throw new HeDeviceError('HE_KEY_MISSING', `no Galois key for element ${gal}`);
    let idx = this.permutations.get(gal);
    if (!idx) {
      idx = this.upload(automorphismIndex(this.logN, gal), `automorphism-${gal}`);
      this.permutations.set(gal, idx);
    }
    const qn = this.qCount * this.n;
    this.keySwitch(a.buf, a.cs, l, key);
    this.ew('ew_add', { a: this.t.ks, b: a.buf, o: this.t.tmp }, { limbs: l + 1 });
    this.ew('ew_permute', { a: this.t.tmp, o: out.buf, c: idx }, { limbs: l + 1 });
    this.ew('ew_permute', { a: this.t.ks, o: out.buf, c: idx }, { limbs: l + 1, aOff: qn, oOff: out.cs });
  }

  rescale(out, a, l) {
    const n = this.n;
    const comps = a.deg + 1;
    const qn = this.qCount * n;
    for (let c = 0; c < comps; c++) {
      this.ew('ew_copy', { a: a.buf, o: this.t.last }, { limbs: 1, modBase: l, aOff: c * a.cs + l * n, oOff: c * n });
    }
    this.intt(this.t.last, 0, comps, l, true);
    this.dispatch('rescale_prep', { a: this.t.last, o: this.t.rs },
      { limbs: l, mod_base: 0, a_stride: n, o_stride: qn, log_t: l }, ceilDiv(l * n, WG), comps);
    for (let c = 0; c < comps; c++) this.ntt(this.t.rs, c * qn, l, 0);
    this.ew('ew_sub_smul', { a: a.buf, b: this.t.rs, o: out.buf, c: this.tablesBuf },
      { limbs: l, aStride: a.cs, bStride: qn, oStride: out.cs, cOff: this.tables.offsets.rescale + l * this.qCount }, comps);
  }

  // --- keys and ciphertexts ---

  // A raw-u32 key (see gpu_codec.go). Residues are checked, uploaded and converted to Montgomery form on the GPU.
  loadKey(raw) {
    const v = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    if (raw.byteLength < RAW_KEY_HEADER || v.getUint32(0, true) !== RAW_KEY_MAGIC) {
      throw new HeDeviceError('HE_KEY_FORMAT', 'not a raw-u32 key');
    }
    const kind = v.getUint32(4, true);
    const gal = Number(v.getBigUint64(8, true));
    const digits = Math.ceil(this.qCount / this.pCount);
    if (v.getUint32(16, true) !== this.qCount - 1 || v.getUint32(20, true) !== this.pCount - 1 || v.getUint32(24, true) !== digits
      || v.getUint32(28, true) !== this.logN || (kind === 0) !== (gal === 0) || kind > 1) {
      throw new HeDeviceError('HE_KEY_FORMAT', 'raw key header does not match the profile');
    }
    const QP = this.qCount + this.pCount;
    const words = digits * 2 * QP * this.n;
    if (raw.byteLength !== RAW_KEY_HEADER + 4 * words) throw new HeDeviceError('HE_KEY_FORMAT', 'raw key has the wrong length');
    const body = new Uint32Array(raw.buffer.slice(raw.byteOffset + RAW_KEY_HEADER, raw.byteOffset + raw.byteLength));
    this.#checkResidues(body, digits * 2, QP);
    const buf = this.upload(body, kind === 0 ? 'key-relin' : `key-galois-${gal}`);
    this.stats.keyBytes += buf.size;
    this.ew('ew_to_mont', { o: buf }, { limbs: QP, oStride: QP * this.n }, digits * 2);
    const key = { buf, kind, gal };
    if (kind === 0) this.keys.relin = key;
    else this.keys.galois.set(gal, key);
    return key;
  }

  #checkResidues(words, polys, limbs) {
    const n = this.n;
    for (let p = 0; p < polys; p++) {
      for (let i = 0; i < limbs; i++) {
        const q = this.moduli[i];
        const base = (p * limbs + i) * n;
        for (let j = 0; j < n; j++) {
          if (words[base + j] >= q) throw new HeDeviceError('HE_RESIDUE', `residue ${j} of limb ${i} is not below its modulus`);
        }
      }
    }
  }

  // A raw-u32 ciphertext; returns a GPU register.
  uploadCt(raw, { degree, level } = {}) {
    const v = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    if (raw.byteLength < RAW_CT_HEADER || v.getUint32(0, true) !== RAW_CT_MAGIC) {
      throw new HeDeviceError('HE_CT_FORMAT', 'not a raw-u32 ciphertext');
    }
    const deg = v.getUint32(4, true);
    const lvl = v.getUint32(8, true);
    if (v.getUint32(12, true) !== this.logN || v.getUint32(16, true) !== this.logSlots || deg < 1 || deg > 2 || lvl >= this.qCount
      || (degree !== undefined && deg !== degree) || (level !== undefined && lvl !== level)) {
      throw new HeDeviceError('HE_CT_FORMAT', 'raw ciphertext header does not match the pinned shape');
    }
    const words = (deg + 1) * (lvl + 1) * this.n;
    if (raw.byteLength !== RAW_CT_HEADER + 4 * words) throw new HeDeviceError('HE_CT_FORMAT', 'raw ciphertext has the wrong length');
    const body = new Uint32Array(raw.buffer.slice(raw.byteOffset + RAW_CT_HEADER, raw.byteOffset + raw.byteLength));
    this.#checkResidues(body, deg + 1, lvl + 1);
    const ct = this.allocCt(deg, lvl);
    this.device.queue.writeBuffer(ct.buf, 0, body.buffer, 0, body.byteLength);
    this.stats.uploadBytes += body.byteLength;
    return { ...ct, scale: v.getFloat64(24, true) };
  }

  async readCt(ct, scale) {
    this.flush();
    const bytes = (ct.deg + 1) * ct.cs * 4;
    const staging = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(ct.buf, 0, staging, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Uint8Array(RAW_CT_HEADER + bytes);
    const v = new DataView(out.buffer);
    [RAW_CT_MAGIC, ct.deg, ct.level, this.logN, this.logSlots, 0].forEach((w, i) => v.setUint32(4 * i, w, true));
    v.setFloat64(24, scale, true);
    out.set(new Uint8Array(staging.getMappedRange()), RAW_CT_HEADER);
    staging.unmap();
    staging.destroy();
    this.stats.readbackBytes += bytes;
    return out;
  }

  // --- program ops (rnsOp in gpu_record.go) ---

  execOp(op, a, b, out, blobBuf) {
    const l = op.level;
    const n = this.n;
    const lim = { limbs: l + 1 };
    const copyComps = (src, from, to) => {
      for (let c = from; c <= to; c++) this.ew('ew_copy', { a: src.buf, o: out.buf }, { ...lim, aOff: c * src.cs, oOff: c * out.cs });
    };
    switch (op.op) {
      case 'zero':
        this.clear(out.buf, 0, (out.deg + 1) * out.cs);
        return;
      case 'add':
      case 'sub': {
        const common = Math.min(a.deg, b.deg);
        this.ew(op.op === 'add' ? 'ew_add' : 'ew_sub', { a: a.buf, b: b.buf, o: out.buf },
          { ...lim, aStride: a.cs, bStride: b.cs, oStride: out.cs }, common + 1);
        if (a.deg > common) copyComps(a, common + 1, op.deg);
        for (let c = common + 1; c <= Math.min(b.deg, op.deg); c++) {
          this.ew(op.op === 'add' ? 'ew_copy' : 'ew_neg', { a: b.buf, o: out.buf }, { ...lim, aOff: c * b.cs, oOff: c * out.cs });
        }
        return;
      }
      case 'smul':
        this.ew('ew_smul', { a: a.buf, o: out.buf, c: blobBuf }, { ...lim, aStride: a.cs, oStride: out.cs, cOff: op.k }, a.deg + 1);
        return;
      case 'sadd':
        this.ew('ew_sadd', { a: a.buf, o: out.buf, c: blobBuf }, { ...lim, cOff: op.k });
        copyComps(a, 1, a.deg);
        return;
      case 'smac': {
        const common = Math.min(a.deg, op.deg);
        this.ew('ew_smac', { a: a.buf, b: b.buf, o: out.buf, c: blobBuf },
          { ...lim, aStride: a.cs, bStride: b.cs, oStride: out.cs, cOff: op.k }, common + 1);
        if (op.deg > common) copyComps(b, common + 1, op.deg);
        return;
      }
      case 'pmul':
        this.ew('ew_pmul', { a: a.buf, o: out.buf, c: blobBuf },
          { ...lim, aStride: a.cs, oStride: out.cs, cOff: op.pt, x0: this.logN - 7 }, a.deg + 1);
        return;
      case 'padd':
        this.ew('ew_padd', { a: a.buf, o: out.buf, c: blobBuf }, { ...lim, cOff: op.pt, x0: this.logN - 7 });
        copyComps(a, 1, a.deg);
        return;
      case 'tensor':
        this.ew('ew_tensor', { a: a.buf, b: b.buf, o: out.buf }, { ...lim, aStride: a.cs, bStride: b.cs, oStride: out.cs }, 3);
        return;
      case 'relin':
        this.relin(out, a, l);
        return;
      case 'rescale':
        this.rescale(out, a, l);
        return;
      case 'rotate':
        this.rotate(out, a, op.gal, l);
        return;
      default:
        throw new HeDeviceError('HE_OP', `unknown op ${op.op} (n=${n})`);
    }
  }
}

export function outLevel(op) {
  return op.op === 'rescale' ? op.level - 1 : op.level;
}
