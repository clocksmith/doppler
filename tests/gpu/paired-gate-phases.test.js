// Execute the actual phase bodies over symbolic paired values. This checks
// expression/dataflow identity, not compiler output or physical numerics.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { getKernelConfig } from '../../src/config/kernel-registry-contract.js';
const variants = ['paired_beta', 'paired_decay_scale', 'paired_log_decay', 'paired_decay'];
const phases = [];
for (const variant of variants) {
  const config = getKernelConfig('qwen_linear_attention_prepare', variant);
  const source = await readFile(new URL(`../../src/gpu/kernels/${config.shaderFile}`, import.meta.url), 'utf8');
  const body = source.slice(source.indexOf('fn main(')).slice(source.slice(source.indexOf('fn main(')).indexOf('{') + 1).trim().slice(0, -1);
  const javascript = body.replace(/\b(let|var)\b/g, 'let').replace(/(\d+)u\b/g, '$1').replace(/vec2<f32>/g, 'pair')
    .replace(/= -pair_mul\((.*)\);/g, '= negate(pair_mul($1));');
  phases.push(new Function('gid', 'u', 'a', 'b', 'dt_bias', 'a_log', 'gates', 'pair', 'pair_sigmoid', 'pair_exp', 'pair_softplus', 'pair_add', 'pair_mul', 'negate', 'select', javascript));
}
const fn = name => (...args) => [name, ...args];
const pair = fn('pair'), sigmoid = fn('sigmoid'), exp = fn('exp'), softplus = fn('softplus'), add = fn('add'), mul = fn('mul'), negate = fn('negate');
let cases = 0;
for (const tokens of [1, 3, 52, 184, 256]) for (const heads of [1, 3, 16]) for (const packed of [false, true]) {
  const count = tokens * heads, offset = packed ? count : 0;
  const a = Array.from({ length: count }, (_, i) => ['a', i]);
  const b = Array.from({ length: count + offset }, (_, i) => ['b', i]);
  const bias = Array.from({ length: heads }, (_, i) => ['dt', i]);
  const scale = Array.from({ length: heads }, (_, i) => ['scale', i]);
  const storage = Array(count * 2).fill(undefined), writes = [];
  let phaseIndex = -1, lane = -1;
  const gates = new Proxy(storage, {
    get(target, key) {
      if (!/^\d+$/.test(String(key))) return target[key];
      assert.equal(Number(key), lane * 2 + 1, 'phase may read only its own decay slot');
      assert(phaseIndex >= 2); assert.notEqual(target[key], undefined); return target[key];
    },
    set(target, key, value) {
      assert.equal(Number(key), lane * 2 + (phaseIndex === 0 ? 0 : 1));
      target[key] = value; writes.push([phaseIndex, lane, Number(key)]); return true;
    },
  });
  const u = { num_tokens: tokens, num_v_heads: heads, packed_flags: Number(packed), b_offset: offset };
  for (phaseIndex = 0; phaseIndex < phases.length; phaseIndex++) {
    for (lane = count + 2; lane >= 0; lane--) {
      phases[phaseIndex]({ x: lane }, u, a, b, bias, scale, gates, pair, sigmoid, exp, softplus, add, mul, negate, (no, yes, condition) => condition ? yes : no);
    }
  }
  assert.equal(writes.length, count * 4, 'bounds guards exclude padded lanes');
  for (let i = 0; i < count; i++) {
    assert.deepEqual(storage[i * 2], sigmoid(b[i + offset]));
    assert.deepEqual(storage[i * 2 + 1], exp(negate(mul(exp(pair(scale[i % heads], 0)), softplus(add(a[i], pair(bias[i % heads], 0)))))));
  }
  cases++;
}
console.log(`paired gate phases: ${cases} actual-body symbolic, packed-offset, lane-alias and padded-dispatch cases pass`);
