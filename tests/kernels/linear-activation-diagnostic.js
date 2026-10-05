/** Test-only F32 activation experiment, qualified only on captured operands. */
export function buildLinearActivationDiagnostic(source) {
  const start = source.indexOf('fn silu(x: f32) -> f32 {');
  const end = source.indexOf('\n@compute', start);
  if (start < 0 || end < 0) throw Error('Missing captured SiLU boundary');
  return source.slice(0, start) + `
// Range-reduced Taylor polynomial. FMA keeps the short Horner chain explicit.
// This diagnostic fails with NaN outside its tested normal-result domain.
fn diagnostic_exp(x: f32) -> f32 {
  if (x < -80.0 || x > 80.0) { return bitcast<f32>(bitcast<u32>(x) | 0x7fc00000u); }
  let n = round(x * 1.4426950408889634);
  let r_hi = fma(-n, 0.693145751953125, x);
  let r = fma(-n, 0.000001428606765330187, r_hi);
  var p = 0.0001984126984126984;
  p = fma(p, r, 0.001388888888888889);
  p = fma(p, r, 0.008333333333333333);
  p = fma(p, r, 0.041666666666666664);
  p = fma(p, r, 0.16666666666666666);
  p = fma(p, r, 0.5);
  p = fma(p, r, 1.0);
  p = fma(p, r, 1.0);
  return p * bitcast<f32>(u32(i32(n) + 127) << 23u);
}

fn diagnostic_reciprocal(denominator: f32) -> f32 {
  let estimate = 1.0 / denominator;
  let residual = fma(-denominator, estimate, 1.0);
  return fma(estimate, residual, estimate);
}

fn silu(x: f32) -> f32 {
  let z = diagnostic_exp(-abs(x));
  let reciprocal = diagnostic_reciprocal(1.0 + z);
  let numerator = select(x * z, x, x >= 0.0);
  return numerator * reciprocal;
}
` + source.slice(end);
}
