// Execute the actual phase bodies over symbolic paired values. This checks
// expression/dataflow identity, not compiler output or physical numerics.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { getKernelConfig } from '../../src/config/kernel-registry-contract.js';
const variants = ['paired_beta', 'paired_decay_scale', 'paired_log_decay', 'paired_decay'];
// Dispatch rows per phase, as in src/gpu/kernels/paired-linear-attention.js;
// one extra row checks that the scale phase's row guard drops it.
const rows = [1, 3, 1, 1];
const phases = [];
for (const variant of variants) {
  const config = getKernelConfig('qwen_linear_attention_prepare', variant);
  const source = await readFile(new URL(`../../src/gpu/kernels/${config.shaderFile}`, import.meta.url), 'utf8');
  const body = source.slice(source.indexOf('fn main(')).slice(source.slice(source.indexOf('fn main(')).indexOf('{') + 1).trim().slice(0, -1);
  const javascript = body.replace(/\b(let|var)\b/g, 'let').replace(/(\d+)u\b/g, '$1').replace(/vec2<f32>/g, 'pair')
    .replace(/= -pair_mul\((.*)\);/g, '= negate(pair_mul($1));')
    .replace(/([(,=]\s*)-([A-Za-z_]\w*)\b(?!\s*\()/g, '$1negate($2)');
  phases.push(new Function('gid', 'u', 'a', 'b', 'dt_bias', 'a_log', 'gates', 'pair', 'pair_sigmoid', 'pair_exp', 'pair_log', 'pair_softplus', 'pair_add', 'pair_mul', 'negate', 'select', javascript));
}
// Each symbolic value carries a numeric shadow `x` (the high component) so the
// bodies' sign branches follow real values while the expressions stay symbolic.
const node = (x, ...expression) => Object.assign(expression, { x });
const pair = (high, low) => node(high[0] === undefined ? high : high.x, 'pair', high, low);
const sigmoid = v => node(1 / (1 + Math.exp(-v.x)), 'sigmoid', v), exp = v => node(Math.exp(v.x), 'exp', v);
const log = v => node(Math.log(v.x), 'log', v), softplus = v => node(Math.log1p(Math.exp(v.x)), 'softplus', v);
const add = (l, r) => node(l.x + r.x, 'add', l, r), mul = (l, r) => node(l.x * r.x, 'mul', l, r), negate = v => node(-v.x, 'negate', v);
let cases = 0;
for (const tokens of [1, 3, 52, 184, 256]) for (const heads of [1, 3, 16]) for (const packed of [false, true]) {
  const count = tokens * heads, offset = packed ? count : 0;
  // Mixed signs exercise both softplus branches; bias values are exact in F32.
  const a = Array.from({ length: count }, (_, i) => node((i % 5) - 2.5, 'a', i));
  const b = Array.from({ length: count + offset }, (_, i) => node(i, 'b', i));
  const bias = Array.from({ length: heads }, (_, i) => (i % 3) * 0.75 - 0.5);
  const scale = Array.from({ length: heads }, (_, i) => -1 - i / 8);
  // Gate storage: count pairs of (beta, decay) followed by count scale-phase scratch values.
  const storage = Array(count * 3).fill(undefined), writes = [];
  let phaseIndex = -1, lane = -1, row = -1;
  const decay = () => lane * 2 + 1, scratch = () => count * 2 + lane;
  const gates = new Proxy(storage, {
    get(target, key) {
      if (!/^\d+$/.test(String(key))) return target[key];
      assert(phaseIndex >= 2, 'only the log-decay and decay phases read gates');
      assert([decay(), ...(phaseIndex === 2 ? [scratch()] : [])].includes(Number(key)), 'phase may read only its own decay and scratch slots');
      assert.notEqual(target[key], undefined); return target[key];
    },
    set(target, key, value) {
      const expected = phaseIndex === 0 ? lane * 2 : phaseIndex === 1 && row === 1 ? scratch() : decay();
      assert.equal(Number(key), expected);
      target[key] = value; writes.push([phaseIndex, lane, row, Number(key)]); return true;
    },
  });
  const u = { num_tokens: tokens, num_v_heads: heads, packed_flags: Number(packed), b_offset: offset };
  for (phaseIndex = 0; phaseIndex < phases.length; phaseIndex++) {
    for (row = 0; row < rows[phaseIndex]; row++) {
      for (lane = count + 2; lane >= 0; lane--) {
        phases[phaseIndex]({ x: lane, y: row }, u, a, b, bias, scale, gates, pair, sigmoid, exp, log, softplus, add, mul, negate, (no, yes, condition) => condition ? yes : no);
      }
    }
  }
  assert.equal(writes.length, count * 5, 'bounds and row guards exclude padded lanes and rows');
  for (let i = 0; i < count; i++) {
    const head = i % heads, input = add(a[i], pair(bias[head], 0));
    const positive = input.x > 0;
    const stable = add(pair(1, 0), exp(positive ? negate(input) : input));
    const softplusValue = positive ? add(input, log(stable)) : log(stable);
    assert.deepEqual(storage[i * 2], sigmoid(b[i + offset]));
    assert.deepEqual(storage[count * 2 + i], stable);
    assert.deepEqual(storage[i * 2 + 1], exp(negate(mul(exp(pair(scale[head], 0)), softplusValue))));
  }
  cases++;
}
console.log(`paired gate phases: ${cases} actual-body symbolic, packed-offset, lane-alias, sign-branch and padded-dispatch cases pass`);
