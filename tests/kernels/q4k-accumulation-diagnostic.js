/** F32-only numerical experiments. Never imported by the production runtime. */
export function buildQ4KAccumulationDiagnostic(source, candidate) {
  const original = 'results[m] = results[m] + dot(a_even, w_even) + dot(a_odd, w_odd);';
  if (!source.includes(original)) throw Error('Captured WideTile accumulation boundary changed');
  if (candidate === 'fma-pair-dot') {
    return source.replace(original,
      'results[m] = results[m] + ordered_dot(a_even, w_even) + ordered_dot(a_odd, w_odd);') + `
fn ordered_dot(a: vec4<f32>, b: vec4<f32>) -> f32 {
    let first = fma(a.x, b.x, a.y * b.y);
    let second = fma(a.z, b.z, a.w * b.w);
    return first + second;
}
`;
  }
  if (candidate === 'fma-serial') {
    return source.replace(original, ['even', 'odd'].flatMap(pair =>
      ['x', 'y', 'z', 'w'].map(c => `results[m] = fma(a_${pair}.${c}, w_${pair}.${c}, results[m]);`)).join('\n'));
  }
  if (['fma-four-lanes', 'fma-four-lanes-scaled', 'fma-four-lanes-loop'].includes(candidate)) {
    const reduction = candidate === 'fma-four-lanes-scaled'
      ? 'fma(accum.x, u.alpha, fma(accum.y, u.alpha, fma(accum.z, u.alpha, accum.w * u.alpha)))'
      : candidate === 'fma-four-lanes-loop'
        ? 'reduced * u.alpha'
        : '((accum.x + accum.y) + (accum.z + accum.w)) * u.alpha';
    const reduceLoop = candidate === 'fma-four-lanes-loop'
      ? 'var reduced = 0.0;\nfor (var lane = 0u; lane < 4u; lane++) { reduced = reduced + accum[lane]; }\n'
      : '';
    return source.replace('var results: array<f32, MAX_TILE_M>;', 'var results: array<vec4<f32>, MAX_TILE_M>;')
      .replace('results[i] = 0.0;', 'results[i] = vec4<f32>(0.0);')
      .replace(original, 'results[m] = fma(a_even, w_even, results[m]);\nresults[m] = fma(a_odd, w_odd, results[m]);')
      .replace('C[row * u.N + col] = results[m] * u.alpha;',
        `let accum = results[m];\n${reduceLoop}C[row * u.N + col] = ${reduction};`);
  }
  throw Error(`Unknown accumulation diagnostic: ${candidate}`);
}
