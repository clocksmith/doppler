// CPU expression-trace equivalence, not a GPU arithmetic qualification.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
const source = await readFile(new URL('../../src/gpu/kernels/codegen/includes/paired-elementary.wgsl.inc', import.meta.url), 'utf8');
function functionBody(text, name) {
  const start = text.indexOf(`fn ${name}(`);
  assert(start >= 0);
  const open = text.indexOf('{', start);
  let depth = 1, end = open + 1;
  for (; depth && end < text.length; end++) {
    if (text[end] === '{') depth++;
    if (text[end] === '}') depth--;
  }
  assert.equal(depth, 0);
  return text.slice(open + 1, end - 1);
}
const previous = `
    let one = vec2<f32>(1.0, 0.0);
    if (a.x >= 0.0) { return pair_div(one, pair_add(one, pair_exp(-a))); }
    let exponential = pair_exp(a);
    return pair_div(exponential, pair_add(one, exponential));
`;
const current = functionBody(source, 'pair_sigmoid');
const bits = value => new Uint32Array(new Float32Array([value]).buffer)[0];
function evaluate(body, input) {
  const calls = [];
  const value = v => v && Object.hasOwn(v, 'x') ? ['pair', bits(v.x), bits(v.y)] : v;
  const operation = name => (...args) => {
    const result = [name, ...args.map(value)];
    calls.push(result);
    return result;
  };
  // This deliberately supports only the helper's scalar branch/let/return
  // syntax. Arithmetic leaves are symbolic, so no F64 approximation replaces
  // the paired operations whose evaluation order must remain identical.
  const javascript = body.replaceAll('vec2<f32>', 'pair').replaceAll('-a', 'negate(a)');
  const run = new Function('a', 'pair', 'negate', 'select', 'pair_exp', 'pair_add', 'pair_div', javascript);
  const result = run(input, (x, y) => ({ x, y }), a => ({ x: -a.x, y: -a.y }),
    (a, b, condition) => condition ? b : a, operation('exp'), operation('add'), operation('div'));
  return { calls, result };
}
const edge = [0, -0, 2 ** -149, -(2 ** -149), 2 ** -126, -(2 ** -126),
  1, -1, 40, -40, Infinity, -Infinity, NaN];
let comparisons = 0;
for (const x of edge) for (const y of edge) {
  assert.deepEqual(evaluate(current, { x, y }), evaluate(previous, { x, y }), `Trace differs at ${x},${y}`);
  comparisons++;
}
let random = 624901;
for (let i = 0; i < 4096; i++) {
  random ^= random << 13; random ^= random >>> 17; random ^= random << 5;
  const x = Math.fround(((random >>> 0) / 2 ** 32 - 0.5) * 80);
  const y = Math.fround(x * 2 ** -26);
  assert.deepEqual(evaluate(current, { x, y }), evaluate(previous, { x, y }));
  comparisons++;
}
assert.equal((current.match(/pair_exp\(/g) ?? []).length, 1,
  'Sigmoid must have one static exponential callsite to avoid duplicate compiler expansion.');
console.log(`paired sigmoid preserved ${comparisons} branch expression traces; one exponential callsite`);
