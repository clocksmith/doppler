// Isolated arithmetic experiment. Constants are generic analytic coefficients,
// never fitted to captured outputs or model-specific expected values.
import { readFile, writeFile } from 'node:fs/promises';
const canonical = await readFile(new URL('./frequency-precompute-input.wgsl', import.meta.url), 'utf8');
const norm = await readFile(new URL('./candidate-accurate-qknorm.wgsl', import.meta.url), 'utf8');
const residuals = norm.slice(norm.indexOf('fn add_u64'), norm.indexOf('@compute'));
const literal = x => Number.isInteger(x) ? `${x}.0` : String(x);
const pair = x => `vec2<f32>(${literal(Math.fround(x))}, ${literal(Math.fround(x - Math.fround(x)))})`;
let factorial = 1;
const coefficients = [1];
for (let i = 1; i <= 14; i++) { factorial *= i; coefficients.push(1 / factorial); }
const helpers = residuals + `
// A compensated F32 temporary is reduced to the original F32 table layout.
fn pair_normalize(high: f32, low: f32) -> vec2<f32> {
  let rounded = high + low;
  return vec2<f32>(rounded, addition_residual(high, low, rounded));
}
fn pair_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  let high = a.x + b.x;
  return pair_normalize(high, addition_residual(a.x, b.x, high) + (a.y + b.y));
}
fn pair_multiply(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  let high = a.x * b.x;
  let low = product_residual(a.x, b.x, high) + (a.x * b.y + a.y * b.x) + a.y * b.y;
  return pair_normalize(high, low);
}
fn pair_divide(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  let high = a.x / b.x;
  let remainder = pair_add(a, -pair_multiply(b, vec2<f32>(high, 0.0)));
  return pair_normalize(high, (remainder.x + remainder.y) / b.x);
}
fn logarithm_pair(value: f32) -> vec2<f32> {
  let bits = bitcast<u32>(value);
  var exponent = i32((bits >> 23u) & 255u) - 127;
  var mantissa = bitcast<f32>((bits & 0x7fffffu) | 0x3f800000u);
  if (mantissa > 1.4142135623730951) { mantissa *= 0.5; exponent += 1; }
  let s = pair_divide(pair_add(vec2<f32>(mantissa, 0.0), vec2<f32>(-1.0, 0.0)),
                            pair_add(vec2<f32>(mantissa, 0.0), vec2<f32>(1.0, 0.0)));
  let squared = pair_multiply(s, s);
  var polynomial = ${pair(1 / 27)};
  ${Array.from({ length: 13 }, (_, i) => 25 - i * 2).map(d => `polynomial = pair_add(pair_multiply(polynomial, squared), ${pair(1 / d)});`).join('\n  ')}
  let reduced = pair_multiply(vec2<f32>(2.0, 0.0), pair_multiply(s, polynomial));
  return pair_add(pair_multiply(vec2<f32>(f32(exponent), 0.0), ${pair(Math.LN2)}), reduced);
}
fn exponential_pair(value: vec2<f32>) -> vec2<f32> {
  let exponent = round(value.x * 1.4426950408889634);
  let reduced = pair_add(value, -pair_multiply(vec2<f32>(exponent, 0.0), ${pair(Math.LN2)}));
  var polynomial = ${pair(coefficients[14])};
  ${coefficients.slice(0, 14).reverse().map(c => `polynomial = pair_add(pair_multiply(polynomial, reduced), ${pair(c)});`).join('\n  ')}
  return vec2<f32>(ldexp(polynomial.x, i32(exponent)), ldexp(polynomial.y, i32(exponent)));
}
fn compensated_phase(theta: f32, dimension: u32, base_dimension: u32, position: f32) -> vec2<f32> {
  let exponent = pair_divide(vec2<f32>(-f32(dimension * 2u), 0.0), vec2<f32>(f32(base_dimension), 0.0));
  let frequency = exponential_pair(pair_multiply(exponent, logarithm_pair(theta)));
  let angle = pair_multiply(vec2<f32>(position, 0.0), frequency);
  let phase = sincos_refined(angle.x);
  // In the tested range the neglected quadratic term is below F32 rounding.
  return vec2<f32>(phase.x - phase.y * angle.y, phase.y + phase.x * angle.y);
}
`;
const source = canonical.replace('@compute', helpers + '\n@compute')
  .replace('    let phase = sincos_refined(angle);', `    var phase = sincos_refined(angle);
    // Preserve declared scaled/inverse-frequency paths outside this experiment.
    if (!USE_INVERSE_FREQUENCIES && u.scaling_type == 0u && scale == 1.0
      && u.theta >= 1.0 && u.theta <= 1.0e30 && rope_position <= 16384.0) {
      phase = compensated_phase(u.theta, dimension, u.frequency_base_dim, rope_position);
    }`);
await writeFile(new URL('./candidate-frequency-residual-rope.wgsl', import.meta.url), source);
