#!/usr/bin/env node
/**
 * Null share operator, for measuring the requester alone at a model's shape (see synthetic.js): it speaks the
 * operator protocol for a synthetic plan and answers each linear round with an output share of the right size taken
 * from a fixed pool, computing nothing. A stage's two null operators draw from pools whose sum decodes to values in
 * [-1, 1], so the requester's range checks and nonlinear steps see ordinary numbers.
 *
 *   node null-operator.js --shape 27b --stages 2 --stage S --share 0|1 --listen HOST:PORT   (token in SS_TOKEN)
 */
import net from 'node:net';
import { parseArgs } from 'node:util';
import { syntheticConfig, syntheticPlan } from './synthetic.js';

const { values: args } = parseArgs({ options: { shape: { type: 'string' }, stages: { type: 'string', default: '2' }, stage: { type: 'string' },
  share: { type: 'string' }, listen: { type: 'string' } } });
const config = syntheticConfig(args.shape);
const plan = syntheticPlan(config, Number(args.stages));
const stageIndex = Number(args.stage), share = Number(args.share), token = process.env.SS_TOKEN;
const stage = plan.stages[stageIndex];
const groups = new Map(stage.groups.map(g => [g.id, { in: g.in, out: g.tensors.reduce((s, t) => s + t.shape[0], 0) }]));
// Enough for a 160-column prefill of every group except the head, which carries only the positions whose logits are kept.
const maxOut = Math.max(16 * (groups.get('head')?.out ?? 0), ...[...groups].filter(([id]) => id !== 'head').map(([, g]) => 160 * g.out));
const MAGIC = 0x31535352;

// Pools: share 0 gets uniform words from a fixed LCG; share 1 gets v - share0 with v in [-1, 1] * 2^44, per element.
let s = 0x2545f491n;
const next = () => (s = (s * 6364136223846793005n + 1442695040888963407n) & ((1n << 64n) - 1n));
const POOL = 1 << 16;
const pool = new BigUint64Array(POOL);
for (let i = 0; i < POOL; i++) {
  const r = next(), v = BigInt.asUintN(64, BigInt(Math.round((Number(next() >> 11n) / 2 ** 53 * 2 - 1) * 2 ** 44)));
  pool[i] = share === 0 ? r : BigInt.asUintN(64, v - r);
}
const full = new Uint8Array(maxOut * 8);
for (let i = 0; i < maxOut; i++) new DataView(full.buffer).setBigUint64(8 * i, pool[i % POOL], true);
const packed = new Map();
const payload = (count, shift) => {
  if (!packed.has(shift)) {
    const width = 8 - shift / 8, out = new Uint8Array(width * maxOut);
    for (let i = 0, o = 0; i < maxOut; i++) for (let k = shift / 8; k < 8; k++) out[o++] = full[8 * i + k];
    packed.set(shift, out);
  }
  if (count > maxOut) throw Error('round larger than the pool');
  return packed.get(shift).subarray(0, count * (8 - shift / 8));
};

function frame(header, body = new Uint8Array(0)) {
  const js = Buffer.from(JSON.stringify(header));
  const head = Buffer.alloc(16 + js.length);
  head.writeUInt32LE(MAGIC, 0); head.writeUInt32LE(js.length, 4); js.copy(head, 8); head.writeBigUInt64LE(BigInt(body.length), 8 + js.length);
  return [head, body];
}

net.createServer({ noDelay: true }, socket => {
  let buffered = Buffer.alloc(0), shift = 0;
  socket.on('data', chunk => {
    buffered = Buffer.concat([buffered, chunk]);
    for (;;) {
      if (buffered.length < 8) return;
      const jsLen = buffered.readUInt32LE(4);
      if (buffered.length < 16 + jsLen) return;
      const plen = Number(buffered.readBigUInt64LE(8 + jsLen));
      if (buffered.length < 16 + jsLen + plen) return;
      const h = JSON.parse(buffered.subarray(8, 8 + jsLen));
      buffered = buffered.subarray(16 + jsLen + plen);
      let reply = [{ ok: true }];
      if (h.op === 'hello') reply = frame({ stage: stageIndex, share, fx: plan.fx, fw: plan.fw, planSha256: 'synthetic', caps: ['returnShift', 'prg:aes256-ctr'] });
      else if (h.op === 'options') { shift = h.returnShift ?? 0; reply = frame({ ok: true }); }
      else if (h.op === 'linear') reply = frame({ round: h.round, computeNs: 0 }, payload(h.n * groups.get(h.group).out, shift));
      else if (h.op === 'close' || h.op === 'exit') { const [a, b] = frame({ stats: {} }); socket.write(a); socket.end(b); if (h.op === 'exit') process.exit(0); return; }
      else reply = frame({ ok: true });
      socket.write(reply[0]); socket.write(reply[1]);
    }
  });
  socket.on('error', () => {});
}).listen(Number(args.listen.split(':')[1]), args.listen.split(':')[0], () => process.stdout.write(`READY ${args.listen}\n`));
