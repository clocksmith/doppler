import assert from 'node:assert/strict';

/** Test-only F32 compensated dot products. No precision or model-policy change.
 * Acceptance requires physical reference improvement and unchanged model gates. */
export function buildRecurrentAccumulationDiagnostic(source, scope = 'both', fused = false) {
  assert(['memory', 'readout', 'both'].includes(scope), 'Unknown recurrent accumulation scope');
  const replace = (from, to) => {
    assert.equal(source.split(from).length, 2, `Missing recurrent dot boundary: ${from}`);
    source = source.replace(from, to);
  };
  if (scope !== 'readout') {
    replace('var kv_mem = 0.0;', 'var kv_mem = 0.0;\n    var memory_error = 0.0;');
    replace('kv_mem = fma(recurrent_state[state_idx], k_normed, kv_mem);',
    `let term = recurrent_state[state_idx] * k_normed;
        let next = kv_mem + term;
        let residual = select((term - next) + kv_mem, (kv_mem - next) + term, abs(kv_mem) >= abs(term));
        memory_error += residual + fma(recurrent_state[state_idx], k_normed, -term);
        kv_mem = next;`);
    const delta = fused ? 'let delta = multiply_ordered(v_val - kv_mem, beta);'
      : 'let delta = multiply_ordered(conv_out[conv_row_base + v_base + vd] - kv_mem, beta);';
    replace(delta, `kv_mem += memory_error;\n      ${delta}`);
  }
  if (scope !== 'memory') {
    replace('var out_value = 0.0;', 'var out_value = 0.0;\n    var output_error = 0.0;');
    replace('out_value = fma(recurrent_state[state_idx], q_normed, out_value);',
    `let term = recurrent_state[state_idx] * q_normed;
        let next = out_value + term;
        let residual = select((term - next) + out_value, (out_value - next) + term, abs(out_value) >= abs(term));
        output_error += residual + fma(recurrent_state[state_idx], q_normed, -term);
        out_value = next;`);
    const boundary = fused ? 'let out_sum_sq = reduce_sum('
      : 'output[out_row_base + vd] = out_value;';
    replace(boundary, `out_value += output_error;\n      ${boundary}`);
  }
  return source;
}
