// Go's float64 math as the removed Go tools computed it on arm64 (Go 1.26): math.Exp (the arm64 assembly archExp,
// with its fused multiply-adds) and math.tanh, plus the plan's GELU with the multiply-add the Go compiler fused.
// The compiler evaluates the approximated functions only at the Chebyshev nodes, so matching these bit for bit makes
// the JavaScript program constants equal Lattigo's. Other code uses plain JavaScript math.

import { BigFloat, pi } from './bigfloat.js';

// fma(a, b, c) = a * b + c with one rounding (exact product, then a correctly rounded sum).
export function fma(a, b, c) {
  const p = new BigFloat(128).mul(new BigFloat().setFloat64(a), new BigFloat().setFloat64(b));
  return new BigFloat(53).add(p, new BigFloat().setFloat64(c)).toNumber();
}

const LN2_HI = 6.93147180369123816490e-01;
const LN2_LO = 1.90821492927058770002e-10;
const LOG2E = 1.44269504088896338700e+00;
const P1 = 1.66666666666666657415e-01;
const P2 = -2.77777777770155933842e-03;
const P3 = 6.61375632143793436117e-05;
const P4 = -1.65339022054652515390e-06;
const P5 = 4.13813679705723846039e-08;

// math.archExp (src/math/exp_arm64.s), instruction for instruction.
export function goExp(x) {
  if (Number.isNaN(x)) return x;
  if (x > 7.09782712893383973096e+02) return Infinity;
  if (x < -7.45133219101941108420e+02) return 0;
  if (Math.abs(x) < 2 ** -28) return 1 + x;
  const kf = x < 0 ? fma(LOG2E, x, -0.5) : fma(LOG2E, x, 0.5);
  const k = Math.trunc(kf);
  const hi = fma(-k, LN2_HI, x); // x - k*Ln2Hi
  const lo = k * LN2_LO;
  const r = hi - lo;
  const t = r * r;
  let c = fma(t, P5, P4);
  c = fma(t, c, P3);
  c = fma(t, c, P2);
  c = fma(t, c, P1);
  c = fma(-t, c, r); // r - t*c
  const y = 1 - ((lo - (r * c) / (2 - c)) - hi);
  return y * 2 ** k;
}

const TANH_P = [-9.64399179425052238628e-1, -9.92877231001918586564e1, -1.61468768441708447952e3];
const TANH_Q = [1.12811678491632931402e2, 2.23548839060100448583e3, 4.84406305325125486048e3];

// math.tanh (src/math/tanh.go) with the multiply-adds the arm64 compiler fuses.
export function goTanh(x) {
  const MAXLOG = 8.8029691931113054295988e+01;
  const z = Math.abs(x);
  if (z > 0.5 * MAXLOG) return x < 0 ? -1 : 1;
  if (z >= 0.625) {
    const s = goExp(z + z);
    const r = 1 - 2 / (s + 1);
    return x < 0 ? -r : r;
  }
  if (x === 0) return x;
  const s = x * x;
  const num = (x * s) * fma(s, fma(s, TANH_P[0], TANH_P[1]), TANH_P[2]);
  const den = fma(s, fma(s, fma(x, x, TANH_Q[0]), TANH_Q[1]), TANH_Q[2]);
  return x + num / den;
}

// math.Sqrt(2/math.Pi) as the Go compiler folds it: the exact quotient rounded to float64, then its square root.
export const SQRT_2_OVER_PI = Math.sqrt(new BigFloat(53).quo(new BigFloat(300).setInt64(2), pi(300)).toNumber());

// geluTanh (plan.go) as compiled: 0.5x * (1 + tanh(c * fma(0.044715*x*x, x, x))).
export function goGeluTanh(x) {
  const inner = fma(0.044715 * x * x, x, x);
  return (0.5 * x) * (1 + goTanh(SQRT_2_OVER_PI * inner));
}
